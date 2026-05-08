package ds4

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/gpu"
)

// CUDAEngine holds CUDA GPU state for DS4 inference acceleration.
type CUDAEngine struct {
	ready      bool
	weightPtrs map[string]gpu.CUdeviceptr // raw byte buffers on GPU
	actBuf     *gpu.Buffer                // [maxDim] float32 activation
	outBuf     *gpu.Buffer                // [maxOutDim] float32 output
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
	ce.ready = false
}
