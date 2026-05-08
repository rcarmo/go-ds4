package ds4

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/gpu"
)

// CUDAEngine holds CUDA GPU state for DS4 inference acceleration.
type CUDAEngine struct {
	ready       bool
	weightPtrs  map[string]gpu.CUdeviceptr
	actBuf      *gpu.Buffer
	outBuf      *gpu.Buffer
	expertPool  *gpu.ExpertPool
	expertCache *gpu.ExpertCache
}

// InitGPU initializes GPU acceleration (CUDA or Vulkan).
func (e *Engine) InitGPU() error {
	// Try CUDA first (NVIDIA)
	if gpu.Init() {
		ce := &CUDAEngine{
			weightPtrs: make(map[string]gpu.CUdeviceptr),
		}
		if gpu.InitCUDAGemvQ8_0() {
			ce.ready = true
			e.GPU = ce
			fmt.Printf("[gpu] CUDA Q8_0 GEMV ready on %s\n", gpu.DeviceName())

			// Init IQ2/Q2K kernels too
			gpu.InitCUDAGemvQ2K()
			gpu.InitCUDASwiGLU()
			// Init IQ2 kernel with grid table
			gridSlice := (*[256 * 128 * 8]int8)(unsafe.Pointer(&iq2xxsSignedGrid[0]))
			gpu.InitCUDAGemvIQ2(gridSlice[:])

			// Allocate expert pool (4 slots for FastExperts, 6 otherwise)
			nSlots := NExpertUsed
			if e.FastExperts {
				nSlots = NExpertUsedFast
			}
			pool, err := gpu.NewExpertPool(nSlots, NEmbd, NFFExp)
			if err == nil {
				ce.expertPool = pool
			} else {
				fmt.Printf("[gpu] expert pool alloc failed: %v\n", err)
			}

			// Allocate expert cache (demand-filled, up to 16 per layer)
			const cachedPerLayer = 16
			cache, err := gpu.NewExpertCache(cachedPerLayer)
			if err == nil {
				ce.expertCache = cache
				fmt.Printf("[gpu] Expert cache ready: %d slots/layer (demand-filled)\n", cachedPerLayer)
			} else {
				fmt.Printf("[gpu] expert cache alloc failed: %v\n", err)
			}

			// Upload Q8_0 weight tensors
			uploaded, totalBytes := e.uploadCUDAWeights(ce)
			fmt.Printf("[gpu] Uploaded %d tensors (%.1f MB) to GPU VRAM\n", uploaded, float64(totalBytes)/(1024*1024))
			return nil
		}
	}

	// Fallback: try Vulkan
	vk := gpu.GPUInit()
	if vk != nil {
		e.GPU = vk
		return nil
	}

	return fmt.Errorf("no GPU available (tried CUDA + Vulkan)")
}

func (e *Engine) uploadCUDAWeights(ce *CUDAEngine) (int, int64) {
	type upload struct {
		name   string
		data   []byte
		outDim int
	}
	var uploads []upload

	// Output head (4096→129280) — biggest single matmul
	uploads = append(uploads, upload{"output.weight", e.Weights.Output, NVocab})

	for il := 0; il < NLayer; il++ {
		l := &e.Weights.Layer[il]
		p := fmt.Sprintf("blk.%d.", il)
		// Only the high-value Q8_0 projections (~2 GB total, fits in 12 GB VRAM)
		uploads = append(uploads,
			upload{p + "attn_q_b.weight", l.AttnQB, NHead * NHeadDim},  // 1024→32768 (34 MB)
			upload{p + "attn_output_b.weight", l.AttnOutputB, NEmbd},   // 1024→4096 (34 MB)
			upload{p + "ffn_down_shexp.weight", l.FfnDownShexp, NEmbd}, // 2048→4096 (8.5 MB)
		)
	}

	uploaded := 0
	var totalBytes int64
	for _, u := range uploads {
		if len(u.data) == 0 || u.outDim < 2048 {
			continue
		}
		var ptr gpu.CUdeviceptr
		if err := gpu.CuMemAllocRaw(&ptr, uint64(len(u.data))); err != nil {
			continue
		}
		gpu.EnsureContext()
		gpu.CuMemcpyHtoDRaw(ptr, unsafe.Pointer(&u.data[0]), uint64(len(u.data)))
		ce.weightPtrs[u.name] = ptr
		uploaded++
		totalBytes += int64(len(u.data))
	}
	return uploaded, totalBytes
}

// GPUReady returns true if GPU acceleration is active.
func (e *Engine) GPUReady() bool {
	if e.GPU == nil {
		return false
	}
	switch g := e.GPU.(type) {
	case *CUDAEngine:
		return g.ready
	case *gpu.GPUEngine:
		return g.Ready()
	}
	return false
}

// gpuMatvecQ8_0 attempts GPU dispatch for a Q8_0 matvec.
func (e *Engine) gpuMatvecQ8_0(out []float32, tensorName string, x []float32, inDim, outDim int) bool {
	if e.GPU == nil {
		return false
	}

	switch g := e.GPU.(type) {
	case *CUDAEngine:
		return g.matvecQ8_0(out, tensorName, x, inDim, outDim)
	case *gpu.GPUEngine:
		return g.MatvecQ8_0GPU(out, tensorName, x, inDim, outDim)
	}
	return false
}

func (ce *CUDAEngine) matvecQ8_0(out []float32, tensorName string, x []float32, inDim, outDim int) bool {
	if !ce.ready {
		return false
	}
	// Only GPU-dispatch for large outputs where kernel speedup exceeds PCIe overhead
	if outDim < 4096 {
		return false
	}
	wtPtr, ok := ce.weightPtrs[tensorName]
	if !ok {
		return false
	}

	// Ensure activation buffer
	if ce.actBuf == nil || ce.actBuf.Size < inDim*4 {
		if ce.actBuf != nil {
			ce.actBuf.Free()
		}
		var err error
		ce.actBuf, err = gpu.Malloc(inDim)
		if err != nil {
			return false
		}
	}
	// Ensure output buffer
	if ce.outBuf == nil || ce.outBuf.Size < outDim*4 {
		if ce.outBuf != nil {
			ce.outBuf.Free()
		}
		var err error
		ce.outBuf, err = gpu.Malloc(outDim)
		if err != nil {
			return false
		}
	}

	ce.actBuf.Upload(x[:inDim])

	nBlocks := inDim / 32
	rowBytes := nBlocks * BlockQ8_0Size

	if err := gpu.CUDAMatvecQ8_0(ce.outBuf, ce.actBuf, wtPtr, inDim, outDim, rowBytes); err != nil {
		return false
	}
	ce.outBuf.Download(out[:outDim])
	return true
}

func (ce *CUDAEngine) Close() {
	if ce == nil {
		return
	}
	for _, ptr := range ce.weightPtrs {
		if ptr != 0 {
			gpu.CuMemFreeRaw(ptr)
		}
	}
	if ce.actBuf != nil {
		ce.actBuf.Free()
	}
	if ce.outBuf != nil {
		ce.outBuf.Free()
	}
	if ce.expertPool != nil {
		ce.expertPool.Free()
	}
	if ce.expertCache != nil {
		ce.expertCache.Free()
	}
	ce.ready = false
}

// gpuExpertForward runs all expert compute on GPU without CPU round-trips.
// Gate → Up → SwiGLU → Down all execute on GPU. Single sync at the end.
func (e *Engine) gpuExpertForward(
	ds *DecodeState, layer *LayerWeights,
	experts []expertScore, il int,
) bool {
	ce, ok := e.GPU.(*CUDAEngine)
	if !ok || !ce.ready || ce.expertCache == nil {
		return false
	}
	if !gpu.CudaGemvIQ2Ready() || !gpu.CudaGemvQ2KReady() || !gpu.CudaSwiGLUReady() {
		return false
	}

	expertIdxs := make([]int, len(experts))
	for i, exp := range experts {
		expertIdxs[i] = exp.idx
	}
	e.cacheExpertsOnDemand(ce, il, expertIdxs)
	for _, exp := range experts {
		if !ce.expertCache.IsCached(il, exp.idx) {
			return false
		}
	}

	actBuf := ce.expertCache.ActBuf(il)
	outBuf := ce.expertCache.OutBuf(il)
	midBuf := ce.expertCache.MidBuf(il)
	gateRowBytes := (NEmbd / 256) * BlockIQ2XXSSize
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / 256) * BlockQ2KSize

	// Single activation upload
	actBuf.Upload(ds.FfnNormed)

	for _, exp := range experts {
		gatePtr, upPtr, downPtr, _ := ce.expertCache.Get(il, exp.idx)

		// All on GPU, no sync between:
		// 1. IQ2 gate: actBuf → outBuf [NFFExp]
		gpu.CUDAMatvecIQ2(outBuf, actBuf, gatePtr, NEmbd, NFFExp, gateRowBytes*NFFExp)
		// 2. IQ2 up: actBuf → midBuf [NFFExp]
		gpu.CUDAMatvecIQ2(midBuf, actBuf, upPtr, NEmbd, NFFExp, upRowBytes*NFFExp)
		// 3. SwiGLU on GPU: outBuf = silu(outBuf) * midBuf
		gpu.CUDASwiGLU(outBuf, outBuf, midBuf, NFFExp)
		// 4. Q2K down: outBuf[NFFExp] → midBuf reused as act → ...
		// Need outBuf as activation for Q2K. But outBuf has NFFExp=2048 floats, actBuf has NEmbd=4096.
		// Use midBuf as the SwiGLU output, feed to Q2K:
		// Actually: after SwiGLU, outBuf has the hidden state [NFFExp].
		// Q2K down: weight[NEmbd, NFFExp] × hidden[NFFExp] → result[NEmbd]
		// We need a separate buffer for Q2K input (outBuf) and output.
		// Reuse actBuf (which we no longer need for this expert):
		gpu.CUDAMatvecQ2K(actBuf, outBuf, downPtr, NFFExp, NEmbd, downRowBytes*NEmbd)

		// Only sync + download at the end of this expert
		gpu.Sync()
		result := ds.ExpertOut[:NEmbd]
		actBuf.Download(result)
		for i := 0; i < NEmbd; i++ {
			ds.RoutedOut[i] += exp.score * result[i]
		}

		// Restore activation for next expert
		actBuf.Upload(ds.FfnNormed)
	}
	return true
}

// loadExpertCache pre-loads expert weights into VRAM on demand.
// Called with actual routed experts — caches them for future reuse.
func (e *Engine) cacheExpertsOnDemand(ce *CUDAEngine, il int, expertIdxs []int) {
	if ce.expertCache == nil {
		return
	}
	gateRowBytes := (NEmbd / 256) * BlockIQ2XXSSize
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / 256) * BlockQ2KSize

	l := &e.Weights.Layer[il]
	if len(l.FfnGateExps) == 0 {
		return
	}
	for _, eidx := range expertIdxs {
		if ce.expertCache.IsCached(il, eidx) {
			continue
		}
		gateBase := l.FfnGateExps[eidx*gateRowBytes*NFFExp : (eidx+1)*gateRowBytes*NFFExp]
		upBase := l.FfnUpExps[eidx*upRowBytes*NFFExp : (eidx+1)*upRowBytes*NFFExp]
		downBase := l.FfnDownExps[eidx*downRowBytes*NEmbd : (eidx+1)*downRowBytes*NEmbd]
		ce.expertCache.LoadExpert(il, eidx, gateBase, upBase, downBase)
	}
}
