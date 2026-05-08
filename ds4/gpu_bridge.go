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
	batchBufs   *gpu.BatchedExpertBufs
	stream      gpu.CUstream
	fusedLayers [NLayer]*gpu.FusedLayerBufs // fused Q8_0 projections per layer
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
			gpu.InitCUDAGemvIQ2Opt()
			gpu.InitCUDAGemvQ2KOpt()
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
				// Allocate batched expert buffers
				bb, berr := gpu.NewBatchedExpertBufs(NExpertUsedFast, NEmbd, NFFExp)
				if berr == nil {
					ce.batchBufs = bb
					if s, err := gpu.StreamCreate(); err == nil {
						ce.stream = s
						fmt.Println("[gpu] CUDA stream created")
					}
					fmt.Println("[gpu] Batched expert buffers ready")
				}
			} else {
				fmt.Printf("[gpu] expert cache alloc failed: %v\n", err)
			}

			// Upload Q8_0 weight tensors
			uploaded, totalBytes := e.uploadCUDAWeights(ce)
			fmt.Printf("[gpu] Uploaded %d tensors (%.1f MB) to GPU VRAM\n", uploaded, float64(totalBytes)/(1024*1024))

			// Create fused Q8_0 projection buffers per layer
			fusedCount := 0
			for il := 0; il < NLayer; il++ {
				l := &e.Weights.Layer[il]
				if len(l.AttnQA) > 0 && len(l.AttnKV) > 0 {
					f, err := gpu.NewFusedLayerBufs(l.AttnQA, l.AttnKV, NEmbd, NLoraQ, NHeadDim)
					if err == nil {
						ce.fusedLayers[il] = f
						fusedCount++
					}
				}
			}
			if fusedCount > 0 {
				fmt.Printf("[gpu] Fused Q8_0 projections: %d layers (attn_q_a+kv \u2192 one dispatch)\n", fusedCount)
			}

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
			upload{p + "attn_q_a.weight", l.AttnQA, NLoraQ},
			upload{p + "attn_q_b.weight", l.AttnQB, NHead * NHeadDim},
			upload{p + "attn_kv.weight", l.AttnKV, NHeadDim},
			upload{p + "attn_output_a.weight", l.AttnOutputA, NLoraO},
			upload{p + "attn_output_b.weight", l.AttnOutputB, NEmbd},
			upload{p + "ffn_gate_shexp.weight", l.FfnGateShexp, NFFExp},
			upload{p + "ffn_up_shexp.weight", l.FfnUpShexp, NFFExp},
			upload{p + "ffn_down_shexp.weight", l.FfnDownShexp, NEmbd},
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
	if outDim < 2048 {
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
		if ce.batchBufs != nil {
			ce.batchBufs.Free()
			for il := range ce.fusedLayers {
				if ce.fusedLayers[il] != nil {
					ce.fusedLayers[il].Free()
				}
			}
			if ce.stream != 0 {
				gpu.StreamDestroy(ce.stream)
			}
		}
	}
	ce.ready = false
}

// gpuExpertForward batches all experts into single kernel launches.
// DtoD copies expert weights contiguously, then one IQ2 launch for all gate rows,
// one IQ2 for all up rows, one SwiGLU, one Q2K for all down rows, single sync.
func (e *Engine) gpuExpertForward(
	ds *DecodeState, layer *LayerWeights,
	experts []expertScore, il int,
) []bool {
	ce, ok := e.GPU.(*CUDAEngine)
	if !ok || !ce.ready || ce.expertCache == nil {
		return nil
	}
	if !gpu.CudaGemvIQ2Ready() || !gpu.CudaGemvQ2KReady() || !gpu.CudaSwiGLUReady() {
		return nil
	}

	nExp := len(experts)
	expertIdxs := make([]int, nExp)
	for i, exp := range experts {
		expertIdxs[i] = exp.idx
	}
	e.cacheExpertsOnDemand(ce, il, expertIdxs)

	// Check how many are cached
	nCached := 0
	for _, exp := range experts {
		if ce.expertCache.IsCached(il, exp.idx) {
			nCached++
		}
	}
	if nCached == 0 {
		return nil
	}

	gateRowBytes := (NEmbd / 256) * BlockIQ2XXSSize * NFFExp
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / 256) * BlockQ2KSize * NEmbd

	actBuf := ce.expertCache.ActBuf(il)
	outBuf := ce.expertCache.OutBuf(il)
	midBuf := ce.expertCache.MidBuf(il)

	actBuf.UploadAsync(ds.FfnNormed, ce.stream)

	gpuHandled := make([]bool, nExp)
	for slot, exp := range experts {
		if !ce.expertCache.IsCached(il, exp.idx) {
			continue
		}
		gatePtr, upPtr, downPtr, _ := ce.expertCache.Get(il, exp.idx)

		gpu.CUDAMatvecIQ2OptStream(outBuf, actBuf, gatePtr, NEmbd, NFFExp, gateRowBytes, ce.stream)
		gpu.CUDAMatvecIQ2OptStream(midBuf, actBuf, upPtr, NEmbd, NFFExp, upRowBytes, ce.stream)
		gpu.CUDASwiGLUStream(outBuf, outBuf, midBuf, NFFExp, ce.stream)
		gpu.CUDAMatvecQ2KOptStream(actBuf, outBuf, downPtr, NFFExp, NEmbd, downRowBytes, ce.stream)

		gpu.StreamSync(ce.stream)
		result := ds.ExpertOut[:NEmbd]
		actBuf.Download(result)
		for i := 0; i < NEmbd; i++ {
			ds.RoutedOut[i] += exp.score * result[i]
		}
		actBuf.UploadAsync(ds.FfnNormed, ce.stream)
		gpuHandled[slot] = true
	}
	return gpuHandled
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

// gpuFusedAttnQAKV dispatches fused attn_q_a + attn_kv on GPU (sync).
func (e *Engine) gpuFusedAttnQAKV(qr, kv, x []float32, il int) bool {
	ce, ok := e.GPU.(*CUDAEngine)
	if !ok || !ce.ready || ce.fusedLayers[il] == nil {
		return false
	}
	fused := ce.fusedLayers[il].AttnQAKV
	fused.DispatchAsync(x, ce.stream)
	results := [][]float32{qr, kv}
	fused.Collect(results, ce.stream)
	return true
}
