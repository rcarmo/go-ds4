package ds4

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/rcarmo/go-ds4/pkg/ds4/internal/gpu"
)

// CUDAEngine holds CUDA GPU state for DS4 inference acceleration.
type CUDAEngine struct {
	ready              bool
	strictMode         bool
	weightPtrs         map[string]gpu.CUdeviceptr
	actBuf             *gpu.Buffer
	outBuf             *gpu.Buffer
	actQPtr            gpu.CUdeviceptr
	actQSize           int
	actScaleBuf        *gpu.Buffer
	expertQ8Ptr        gpu.CUdeviceptr
	expertQ8Size       int
	midQ8Ptr           gpu.CUdeviceptr
	midQ8Size          int
	expertPool         *gpu.ExpertPool
	expertCache        *gpu.ExpertCache
	compactExpertCache *gpu.CompactExpertCache
	batchBufs          *gpu.BatchedExpertBufs
	stream             gpu.CUstream
	fusedLayers        [NLayer]*gpu.FusedLayerBufs
	attnQBuf           *gpu.Buffer // [NHead*NHeadDim] Q vectors
	attnKVBuf          *gpu.Buffer // [NSWA*NHeadDim] KV cache
	attnSinkBuf        *gpu.Buffer // [NHead] sinks
	attnOutBuf         *gpu.Buffer // [NHead*NHeadDim] attention output // fused Q8_0 projections per layer
}

func (ce *CUDAEngine) strict() bool { return ce != nil && ce.strictMode }

// InitGPU initializes GPU acceleration (CUDA or Vulkan).
func (e *Engine) InitGPU() error {
	if metalEngine, err := e.initMetalGPU(); err == nil {
		e.GPU = metalEngine
		return nil
	} else if os.Getenv("DS4_GPU_BACKEND") == "metal" {
		return err
	}

	// Try CUDA first (NVIDIA)
	if gpu.Init() {
		ce := &CUDAEngine{
			strictMode: e.StrictGPU,
			weightPtrs: make(map[string]gpu.CUdeviceptr),
		}
		if gpu.InitCUDAGemvQ8_0Prequant() {
			gpu.InitCUDAGemvQ2KQ8K()
			gridSlice := (*[256 * 128 * 8]int8)(unsafe.Pointer(&iq2xxsSignedGrid[0]))
			gpu.InitCUDAGemvIQ2(gridSlice[:])
			gpu.InitCUDAGemvIQ2Q8K()
			gpu.InitCUDAExpertSwiGLUQ8K()
			ce.ready = true
			e.GPU = ce
			fmt.Printf("[gpu] CUDA Q8_0 prequant parity path ready on %s\n", gpu.DeviceName())

			// Upload Q8_0 weight tensors for the parity-safe dense path.
			uploaded, totalBytes := e.uploadCUDAWeights(ce)
			fmt.Printf("[gpu] Uploaded %d Q8_0 tensors (%.1f MB) to GPU VRAM\n", uploaded, float64(totalBytes)/(1024*1024))

			if !gpuNonParityKernelsAllowed() {
				cachedPerLayer := 16
				if e.StrictGPU {
					// Strict mode validates kernels, not cache hit-rate. Keep expert weights
					// transient to avoid masking kernel bugs with VRAM exhaustion.
					cachedPerLayer = 0
				}
				if cache, err := gpu.NewExpertCache(cachedPerLayer); err == nil {
					ce.expertCache = cache
					if e.StrictGPU {
						if bb, berr := gpu.NewBatchedExpertBufs(NExpertUsed, NEmbd, NFFExp); berr == nil {
							ce.batchBufs = bb
						} else {
							return fmt.Errorf("strict GPU batched expert buffers: %w", berr)
						}
						budget := gpu.CompactExpertCacheBudgetFromEnv(0)
						ce.compactExpertCache = gpu.NewCompactExpertCache(budget)
						fmt.Printf("[gpu] Batched expert buffers ready (Q8_K parity); compact expert cache %.1f MB\n", float64(budget)/(1024*1024))
					} else {
						fmt.Printf("[gpu] Expert cache ready: %d slots/layer (Q8_K parity, demand-filled)\n", cachedPerLayer)
					}
					if s, err := gpu.StreamCreate(); err == nil {
						ce.stream = s
					}
				} else {
					fmt.Printf("[gpu] expert cache alloc failed: %v\n", err)
				}
				fmt.Println("[gpu] legacy fused/Vulkan kernels remain disabled unless DS4_UNSAFE_GPU_NONPARITY=1")
				return nil
			}

			fmt.Println("[gpu] unsafe non-parity CUDA expert/fused kernels enabled by DS4_UNSAFE_GPU_NONPARITY=1")
			// Init legacy IQ2/Q2K kernels too
			gpu.InitCUDAGemvQ2K()
			gpu.InitCUDASwiGLU()
			gpu.InitCUDAGemvIQ2Opt()
			gpu.InitCUDAGemvQ2KOpt()
			gpu.InitCUDAAttn()

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
				e.initAttnGPUBufs(ce)
			}

			return nil
		}
	}

	if !gpuNonParityKernelsAllowed() {
		return fmt.Errorf("no parity-safe CUDA Q8_0 GPU available")
	}

	// Fallback: try Vulkan (legacy/non-parity benchmarking only)
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
			upload{p + "attn_output_a.weight", l.AttnOutputA, NLoraO * NOutGroup},
			upload{p + "attn_output_b.weight", l.AttnOutputB, NEmbd},
			upload{p + "attn_compressor_kv.weight", l.CompressorKV, detectOutDim(l.CompressorKV, NEmbd)},
			upload{p + "attn_compressor_gate.weight", l.CompressorGate, detectOutDim(l.CompressorGate, NEmbd)},
			upload{p + "indexer_attn_q_b.weight", l.IndexerQB, NIndexerHead * NIndexerHeadDim},
			upload{p + "indexer_proj.weight", l.IndexerProj, NIndexerHead},
			upload{p + "indexer_compressor_kv.weight", l.IndexerCompKV, detectOutDim(l.IndexerCompKV, NEmbd)},
			upload{p + "indexer_compressor_gate.weight", l.IndexerCompGate, detectOutDim(l.IndexerCompGate, NEmbd)},
			upload{p + "ffn_gate_shexp.weight", l.FfnGateShexp, NFFExp},
			upload{p + "ffn_up_shexp.weight", l.FfnUpShexp, NFFExp},
			upload{p + "ffn_down_shexp.weight", l.FfnDownShexp, NEmbd},
		)
	}

	uploaded := 0
	var totalBytes int64
	for _, u := range uploads {
		if len(u.data) == 0 || (!e.StrictGPU && u.outDim < 2048) {
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
	if ready, ok := metalGPUReady(e.GPU); ok {
		return ready
	}
	return false
}

// gpuNonParityKernelsAllowed returns true only when explicitly requested.
// Dense CUDA Q8_0 now uses the parity-safe prequantized activation path.
// The legacy fused/Vulkan/routed-expert kernels still consume F32 activations
// and bypass Q8_K expert quantization / exact Q2_K traversal, so keep them
// disabled by default.
func gpuNonParityKernelsAllowed() bool {
	return os.Getenv("DS4_UNSAFE_GPU_NONPARITY") == "1"
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
	if ok, handled := metalGPUMatvecQ8_0(e.GPU, out, tensorName, x, inDim, outDim); handled {
		return ok
	}
	return false
}

func (ce *CUDAEngine) matvecQ8_0(out []float32, tensorName string, x []float32, inDim, outDim int) bool {
	if !ce.ready {
		return false
	}
	// Only GPU-dispatch for large outputs where kernel speedup exceeds PCIe overhead,
	// unless strict mode is explicitly validating GPU coverage/correctness.
	if outDim < 2048 && !ce.strict() {
		return false
	}
	wtPtr, ok := ce.weightPtrs[tensorName]
	if !ok {
		return false
	}
	return ce.matvecQ8_0Ptr(out, wtPtr, x, inDim, outDim)
}

func (ce *CUDAEngine) matvecQ8_0Ptr(out []float32, wtPtr gpu.CUdeviceptr, x []float32, inDim, outDim int) bool {
	nBlocks := (inDim + 31) / 32
	rowBytes := nBlocks * BlockQ8_0Size

	// Ensure prequantized activation buffers.
	xqLen := nBlocks * 32
	if ce.actQPtr == 0 || ce.actQSize < xqLen {
		if ce.actQPtr != 0 {
			gpu.CuMemFreeRaw(ce.actQPtr)
		}
		if err := gpu.CuMemAllocRaw(&ce.actQPtr, uint64(xqLen)); err != nil {
			return false
		}
		ce.actQSize = xqLen
	}
	if ce.actScaleBuf == nil || ce.actScaleBuf.Size < nBlocks*4 {
		if ce.actScaleBuf != nil {
			ce.actScaleBuf.Free()
		}
		var err error
		ce.actScaleBuf, err = gpu.Malloc(nBlocks)
		if err != nil {
			return false
		}
	}
	// Ensure output buffer.
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

	xq := make([]int8, xqLen)
	xscale := make([]float32, nBlocks)
	xsum := make([]float32, nBlocks)
	quantizeQ8_0Activation(x[:inDim], xq, xscale, xsum)
	gpu.CuMemcpyHtoDRaw(ce.actQPtr, unsafe.Pointer(&xq[0]), uint64(len(xq)))
	ce.actScaleBuf.Upload(xscale)

	if err := gpu.CUDAMatvecQ8_0Prequant(ce.outBuf, ce.actQPtr, ce.actScaleBuf.Ptr, wtPtr, inDim, outDim, rowBytes); err != nil {
		return false
	}
	ce.outBuf.Download(out[:outDim])
	return true
}

func (e *Engine) gpuMatvecQ8_0Grouped(out []float32, tensorName string, x []float32, inDim, outDim, groupSize int) bool {
	if ok, handled := metalGPUMatvecQ8_0Grouped(e.GPU, out, tensorName, x, inDim, outDim, groupSize); handled {
		return ok
	}
	ce, ok := e.GPU.(*CUDAEngine)
	if !ok || !ce.ready {
		return false
	}
	basePtr, ok := ce.weightPtrs[tensorName]
	if !ok {
		return false
	}
	nGroups := groupSize
	groupDim := inDim / nGroups
	rank := outDim
	nBlocks := (groupDim + 31) / 32
	rowBytes := nBlocks * BlockQ8_0Size
	for g := 0; g < nGroups; g++ {
		ptr := gpu.CUdeviceptr(uint64(basePtr) + uint64(g*rank*rowBytes))
		gx := x[g*groupDim : (g+1)*groupDim]
		goff := g * rank
		if !ce.matvecQ8_0Ptr(out[goff:goff+rank], ptr, gx, groupDim, rank) {
			return false
		}
	}
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
	if ce.actQPtr != 0 {
		gpu.CuMemFreeRaw(ce.actQPtr)
		ce.actQPtr = 0
	}
	if ce.actScaleBuf != nil {
		ce.actScaleBuf.Free()
	}
	if ce.outBuf != nil {
		ce.outBuf.Free()
	}
	if ce.expertQ8Ptr != 0 {
		gpu.CuMemFreeRaw(ce.expertQ8Ptr)
		ce.expertQ8Ptr = 0
	}
	if ce.midQ8Ptr != 0 {
		gpu.CuMemFreeRaw(ce.midQ8Ptr)
		ce.midQ8Ptr = 0
	}
	if ce.expertPool != nil {
		ce.expertPool.Free()
	}
	if ce.compactExpertCache != nil {
		fmt.Printf("[gpu] %s\n", ce.compactExpertCache.StatsString())
		ce.compactExpertCache.Free()
		ce.compactExpertCache = nil
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
	if handled, ok := metalGPUExpertForward(e.GPU, ds, layer, experts, il); ok {
		return handled
	}
	ce, ok := e.GPU.(*CUDAEngine)
	if !ok || !ce.ready || ce.expertCache == nil {
		return nil
	}
	if !gpu.CudaGemvIQ2Q8KReady() || !gpu.CudaGemvQ2KQ8KReady() {
		return nil
	}

	nExp := len(experts)
	if !e.StrictGPU {
		expertIdxs := make([]int, nExp)
		for i, exp := range experts {
			expertIdxs[i] = exp.idx
		}
		e.cacheExpertsOnDemand(ce, il, expertIdxs)
	}

	gateRowBytes := (NEmbd / 256) * BlockIQ2XXSSize
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / 256) * BlockQ2KSize

	actBuf := ce.expertCache.ActBuf(il) // reused as down output [NEmbd] float32
	outBuf := ce.expertCache.OutBuf(il) // gate [NFFExp] float32
	midBuf := ce.expertCache.MidBuf(il) // up [NFFExp] float32

	inQ8Len := (NEmbd / QK_K) * BlockQ8KSize
	if ce.expertQ8Ptr == 0 || ce.expertQ8Size < inQ8Len {
		if ce.expertQ8Ptr != 0 {
			gpu.CuMemFreeRaw(ce.expertQ8Ptr)
		}
		if err := gpu.CuMemAllocRaw(&ce.expertQ8Ptr, uint64(inQ8Len)); err != nil {
			return nil
		}
		ce.expertQ8Size = inQ8Len
	}
	midQ8Len := (NFFExp / QK_K) * BlockQ8KSize
	if ce.midQ8Ptr == 0 || ce.midQ8Size < midQ8Len {
		if ce.midQ8Ptr != 0 {
			gpu.CuMemFreeRaw(ce.midQ8Ptr)
		}
		if err := gpu.CuMemAllocRaw(&ce.midQ8Ptr, uint64(midQ8Len)); err != nil {
			return nil
		}
		ce.midQ8Size = midQ8Len
	}
	gpu.CuMemcpyHtoDRaw(ce.expertQ8Ptr, unsafe.Pointer(&ds.RoutedXQ[0]), uint64(inQ8Len))

	if e.StrictGPU && ce.batchBufs != nil && gpu.CudaExpertSwiGLUQ8KReady() {
		return e.gpuExpertForwardStrictBatched(ds, layer, experts, il, gateRowBytes, upRowBytes, downRowBytes, inQ8Len, midQ8Len)
	}

	gpuHandled := make([]bool, nExp)
	for slot, exp := range experts {
		var gatePtr, upPtr, downPtr gpu.CUdeviceptr
		var transientPtrs []gpu.CUdeviceptr
		if e.StrictGPU {
			gateBase := layer.FfnGateExps[exp.idx*gateRowBytes*NFFExp : (exp.idx+1)*gateRowBytes*NFFExp]
			upBase := layer.FfnUpExps[exp.idx*upRowBytes*NFFExp : (exp.idx+1)*upRowBytes*NFFExp]
			downBase := layer.FfnDownExps[exp.idx*downRowBytes*NEmbd : (exp.idx+1)*downRowBytes*NEmbd]
			for _, item := range []struct {
				data []byte
				ptr  *gpu.CUdeviceptr
			}{{gateBase, &gatePtr}, {upBase, &upPtr}, {downBase, &downPtr}} {
				if err := gpu.CuMemAllocRaw(item.ptr, uint64(len(item.data))); err != nil {
					for _, ptr := range transientPtrs {
						gpu.CuMemFreeRaw(ptr)
					}
					panic(fmt.Sprintf("strict GPU: failed transient expert alloc layer %d expert %d: %v", il, exp.idx, err))
				}
				transientPtrs = append(transientPtrs, *item.ptr)
				gpu.CuMemcpyHtoDRaw(*item.ptr, unsafe.Pointer(&item.data[0]), uint64(len(item.data)))
			}
		} else {
			if !ce.expertCache.IsCached(il, exp.idx) {
				continue
			}
			gatePtr, upPtr, downPtr, _ = ce.expertCache.Get(il, exp.idx)
		}
		if len(transientPtrs) > 0 {
			defer func(ptrs []gpu.CUdeviceptr) {
				for _, ptr := range ptrs {
					gpu.CuMemFreeRaw(ptr)
				}
			}(transientPtrs)
		}

		if err := gpu.CUDAMatvecIQ2Q8K(outBuf, ce.expertQ8Ptr, gatePtr, NEmbd, NFFExp, gateRowBytes, ce.stream); err != nil {
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: IQ2 gate kernel failed layer %d expert %d: %v", il, exp.idx, err))
			}
			continue
		}
		if err := gpu.CUDAMatvecIQ2Q8K(midBuf, ce.expertQ8Ptr, upPtr, NEmbd, NFFExp, upRowBytes, ce.stream); err != nil {
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: IQ2 up kernel failed layer %d expert %d: %v", il, exp.idx, err))
			}
			continue
		}
		if err := gpu.StreamSyncErr(ce.stream); err != nil {
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: IQ2 gate/up sync failed layer %d expert %d: %v", il, exp.idx, err))
			}
			continue
		}

		gate := ds.ExpertGate[slot*NFFExp : (slot+1)*NFFExp]
		up := ds.ExpertUp[slot*NFFExp : (slot+1)*NFFExp]
		outBuf.Download(gate)
		midBuf.Download(up)

		limit := ds.Cfg().SwiGLUClampExp
		if limit > 1.0e-6 {
			for i := 0; i < NFFExp; i++ {
				if gate[i] > limit {
					gate[i] = limit
				}
				if up[i] > limit {
					up[i] = limit
				} else if up[i] < -limit {
					up[i] = -limit
				}
			}
		}
		swiGLU(gate, gate, up)
		for i := 0; i < NFFExp; i++ {
			gate[i] *= exp.score
		}
		midQ := ds.ExpertMidQ[slot*midQ8Len : (slot+1)*midQ8Len]
		quantizeQ8KPadded(gate, midQ)
		gpu.CuMemcpyHtoDRaw(ce.midQ8Ptr, unsafe.Pointer(&midQ[0]), uint64(midQ8Len))

		if err := gpu.CUDAMatvecQ2KQ8K(actBuf, ce.midQ8Ptr, downPtr, NFFExp, NEmbd, downRowBytes, ce.stream); err != nil {
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: Q2K down kernel failed layer %d expert %d: %v", il, exp.idx, err))
			}
			continue
		}
		if err := gpu.StreamSyncErr(ce.stream); err != nil {
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: Q2K down sync failed layer %d expert %d: %v", il, exp.idx, err))
			}
			continue
		}
		result := ds.ExpertOut[slot*NEmbd : (slot+1)*NEmbd]
		actBuf.Download(result)
		for i := 0; i < NEmbd; i++ {
			ds.RoutedOut[i] += result[i]
		}
		gpuHandled[slot] = true
	}
	return gpuHandled
}

func (e *Engine) gpuExpertForwardStrictBatched(ds *DecodeState, layer *LayerWeights, experts []expertScore, il int, gateRowBytes, upRowBytes, downRowBytes, inQ8Len, midQ8Len int) []bool {
	ce := e.GPU.(*CUDAEngine)
	bb := ce.batchBufs
	nExp := len(experts)
	for slot, exp := range experts {
		gateBase := layer.FfnGateExps[exp.idx*gateRowBytes*NFFExp : (exp.idx+1)*gateRowBytes*NFFExp]
		upBase := layer.FfnUpExps[exp.idx*upRowBytes*NFFExp : (exp.idx+1)*upRowBytes*NFFExp]
		downBase := layer.FfnDownExps[exp.idx*downRowBytes*NEmbd : (exp.idx+1)*downRowBytes*NEmbd]
		gateDst := gpu.CUdeviceptr(uint64(bb.GateBuf) + uint64(slot*len(gateBase)))
		upDst := gpu.CUdeviceptr(uint64(bb.UpBuf) + uint64(slot*len(upBase)))
		downDst := gpu.CUdeviceptr(uint64(bb.DownBuf) + uint64(slot*len(downBase)))
		if ce.compactExpertCache != nil {
			if err := ce.compactExpertCache.CopyTo(gateDst, il, exp.idx, gpu.CompactExpertGate, gateBase); err != nil {
				panic(fmt.Sprintf("strict GPU: compact gate cache layer %d expert %d: %v", il, exp.idx, err))
			}
			if err := ce.compactExpertCache.CopyTo(upDst, il, exp.idx, gpu.CompactExpertUp, upBase); err != nil {
				panic(fmt.Sprintf("strict GPU: compact up cache layer %d expert %d: %v", il, exp.idx, err))
			}
			if err := ce.compactExpertCache.CopyTo(downDst, il, exp.idx, gpu.CompactExpertDown, downBase); err != nil {
				panic(fmt.Sprintf("strict GPU: compact down cache layer %d expert %d: %v", il, exp.idx, err))
			}
		} else {
			gpu.CuMemcpyHtoDRaw(gateDst, unsafe.Pointer(&gateBase[0]), uint64(len(gateBase)))
			gpu.CuMemcpyHtoDRaw(upDst, unsafe.Pointer(&upBase[0]), uint64(len(upBase)))
			gpu.CuMemcpyHtoDRaw(downDst, unsafe.Pointer(&downBase[0]), uint64(len(downBase)))
		}
	}
	if err := gpu.CUDAMatvecIQ2Q8K(bb.GateOut, ce.expertQ8Ptr, bb.GateBuf, NEmbd, nExp*NFFExp, gateRowBytes, ce.stream); err != nil {
		panic(fmt.Sprintf("strict GPU: batched IQ2 gate failed layer %d: %v", il, err))
	}
	if err := gpu.CUDAMatvecIQ2Q8K(bb.UpOut, ce.expertQ8Ptr, bb.UpBuf, NEmbd, nExp*NFFExp, upRowBytes, ce.stream); err != nil {
		panic(fmt.Sprintf("strict GPU: batched IQ2 up failed layer %d: %v", il, err))
	}
	if err := gpu.StreamSyncErr(ce.stream); err != nil {
		panic(fmt.Sprintf("strict GPU: batched IQ2 sync failed layer %d: %v", il, err))
	}
	weights := ds.RouteLogits[:nExp]
	for i, exp := range experts {
		weights[i] = exp.score
	}
	var weightsPtr gpu.CUdeviceptr
	if err := gpu.CuMemAllocRaw(&weightsPtr, uint64(nExp*4)); err != nil {
		panic(fmt.Sprintf("strict GPU: route weight alloc failed layer %d: %v", il, err))
	}
	defer gpu.CuMemFreeRaw(weightsPtr)
	gpu.CuMemcpyHtoDRaw(weightsPtr, unsafe.Pointer(&weights[0]), uint64(nExp*4))
	if ce.midQ8Ptr == 0 || ce.midQ8Size < nExp*midQ8Len {
		if ce.midQ8Ptr != 0 {
			gpu.CuMemFreeRaw(ce.midQ8Ptr)
		}
		if err := gpu.CuMemAllocRaw(&ce.midQ8Ptr, uint64(nExp*midQ8Len)); err != nil {
			panic(fmt.Sprintf("strict GPU: batched midQ alloc failed layer %d: %v", il, err))
		}
		ce.midQ8Size = nExp * midQ8Len
	}
	if err := gpu.CUDAExpertSwiGLUQ8K(bb.GateOut, bb.UpOut, weightsPtr, ce.midQ8Ptr, NFFExp, nExp, ds.Cfg().SwiGLUClampExp, ce.stream); err != nil {
		panic(fmt.Sprintf("strict GPU: batched SwiGLU+Q8K failed layer %d: %v", il, err))
	}
	if err := gpu.StreamSyncErr(ce.stream); err != nil {
		panic(fmt.Sprintf("strict GPU: batched SwiGLU+Q8K sync failed layer %d: %v", il, err))
	}
	gpuHandled := make([]bool, nExp)
	for slot := range experts {
		midPtr := gpu.CUdeviceptr(uint64(ce.midQ8Ptr) + uint64(slot*midQ8Len))
		downPtr := gpu.CUdeviceptr(uint64(bb.DownBuf) + uint64(slot*NEmbd*downRowBytes))
		if err := gpu.CUDAMatvecQ2KQ8K(bb.ActBuf, midPtr, downPtr, NFFExp, NEmbd, downRowBytes, ce.stream); err != nil {
			panic(fmt.Sprintf("strict GPU: batched Q2K down failed layer %d slot %d: %v", il, slot, err))
		}
		if err := gpu.StreamSyncErr(ce.stream); err != nil {
			panic(fmt.Sprintf("strict GPU: batched Q2K down sync failed layer %d slot %d: %v", il, slot, err))
		}
		result := ds.ExpertOut[slot*NEmbd : (slot+1)*NEmbd]
		bb.ActBuf.Download(result)
		for i := 0; i < NEmbd; i++ {
			ds.RoutedOut[i] += result[i]
		}
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
		if err := ce.expertCache.LoadExpert(il, eidx, gateBase, upBase, downBase); err != nil && ce.strict() {
			panic(fmt.Sprintf("strict GPU: failed to cache layer %d expert %d: %v", il, eidx, err))
		}
	}
}

// gpuFusedAttnQAKV dispatches fused attn_q_a + attn_kv on GPU (sync).
func (e *Engine) gpuFusedAttnQAKV(qr, kv, x []float32, il int) bool {
	if !gpuNonParityKernelsAllowed() {
		return false
	}
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

func (e *Engine) initAttnGPUBufs(ce *CUDAEngine) {
	var err error
	ce.attnQBuf, err = gpu.Malloc(NHead * NHeadDim)
	if err != nil {
		return
	}
	ce.attnKVBuf, err = gpu.Malloc(NSWA * NHeadDim)
	if err != nil {
		return
	}
	ce.attnSinkBuf, err = gpu.Malloc(NHead)
	if err != nil {
		return
	}
	ce.attnOutBuf, err = gpu.Malloc(NHead * NHeadDim)
	if err != nil {
		return
	}
	fmt.Println("[gpu] Attention GPU buffers allocated")
}

// gpuAttnScoring runs all 64 heads' attention scoring on GPU.
func (e *Engine) gpuAttnScoring(ds *DecodeState, cache *LayerCache, layer *LayerWeights, il int) bool {
	ce, ok := e.GPU.(*CUDAEngine)
	if !ok || !ce.ready || ce.attnQBuf == nil || !gpu.CudaAttnReady() {
		return false
	}
	nRaw := cache.NRaw
	if nRaw > cache.CapRaw {
		nRaw = cache.CapRaw
	}
	if nRaw == 0 {
		return false
	}

	// Upload Q, KV cache (chronological order), and sinks
	ce.attnQBuf.UploadAsync(ds.Q, ce.stream)

	// Build chronological KV view for GPU
	kvView := make([]float32, nRaw*NHeadDim)
	for t := 0; t < nRaw; t++ {
		row := cache.RawRow(t)
		copy(kvView[t*NHeadDim:(t+1)*NHeadDim], row)
	}
	ce.attnKVBuf.UploadAsync(kvView, ce.stream)

	sinks := tensorF32Unsafe(layer.AttnSinks)
	ce.attnSinkBuf.UploadAsync(sinks[:NHead], ce.stream)

	scale := float32(1.0 / 22.627417) // 1/sqrt(512)

	if err := gpu.CUDABatchedAttn(ce.attnQBuf, ce.attnKVBuf, ce.attnSinkBuf, ce.attnOutBuf,
		nRaw, NHeadDim, scale, ce.stream); err != nil {
		return false
	}
	gpu.StreamSync(ce.stream)
	ce.attnOutBuf.Download(ds.Heads)
	return true
}
