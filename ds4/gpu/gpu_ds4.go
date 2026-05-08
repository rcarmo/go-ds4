package gpu

// DS4 GPU operations — Vulkan compute shaders for DeepSeek V4 Flash inference.
//
// Operations:
//   - MatvecQ8_0:   Q8_0 weight × F32 activation → F32 output
//   - RMSNorm:      in-place RMS normalization with learned scale
//   - Softmax:      in-place softmax over score buffer
//   - VecSiLUMul:   dst = silu(a) * b (element-wise)
//   - Saxpy:        y += alpha * x (attention accumulation)
//
// All shaders use host-visible buffers (simple, correct).
// Weights stay mmap'd in system RAM; we upload activation per call.
// This is the "offload hot matmul" strategy:
//   - Upload x (4096 floats = 16 KB) → GPU
//   - Dispatch GEMV kernel → GPU computes 1024+ dot products in parallel
//   - Download result → CPU continues
//
// For DS4 Q8_0 with F16 scales (34-byte blocks), the GEMV kernel:
//   - Reads weight row from device buffer (pre-uploaded at load)
//   - Each workgroup computes one output row
//   - Per-block: decode F16 scale, dot int8[32] × f32 activation[32], accumulate
//   - 256 threads cooperatively reduce one row of 4096 elements (128 blocks)

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

// GPUEngine holds GPU state for DS4 inference acceleration.
type GPUEngine struct {
	ready   bool
	devName string

	// Persistent weight buffers (uploaded once at load)
	weightBufs map[string]*VkBuf

	// Transient activation/output buffers (reused across calls)
	actBuf *VkBuf // [maxDim] float32
	outBuf *VkBuf // [maxOutDim] float32

	// Compiled kernels
	gemvF32  *VkComputeKernel // F32 weight × F32 activation → F32
	gemvQ8_0 *VkComputeKernel // Q8_0 weight × F32 activation → F32

	mu sync.Mutex
}

// GPUInit attempts to initialize Vulkan GPU compute.
// Returns nil if no GPU is available (CPU fallback transparent).
func GPUInit() *GPUEngine {
	if !VulkanInit() {
		return nil
	}
	g := &GPUEngine{
		ready:      true,
		devName:    vkDevName,
		weightBufs: make(map[string]*VkBuf),
	}

	// Pre-compile kernels from embedded SPIR-V
	// Skip on software renderers (llvmpipe) — they crash on some SPIR-V
	if g.devName != "" && !strings.Contains(g.devName, "llvmpipe") {
		if k, err := VkKernelCreate(SPIRVGemvF32, 3, 8); err == nil {
			g.gemvF32 = k
		}
		if k, err := VkKernelCreate(SPIRVGemvQ8_0F16Scale, 3, 12); err == nil {
			g.gemvQ8_0 = k
		}
	}

	fmt.Printf("[gpu] Vulkan compute ready: %s\n", g.devName)
	if g.gemvF32 != nil {
		fmt.Println("[gpu]   F32 GEMV kernel: ok")
	}
	if g.gemvQ8_0 != nil {
		fmt.Println("[gpu]   Q8_0 GEMV kernel: ok")
	}
	return g
}

// Ready returns true if GPU compute is available.
func (g *GPUEngine) Ready() bool {
	return g != nil && g.ready
}

// DeviceName returns the GPU device name.
func (g *GPUEngine) DeviceName() string {
	if g == nil {
		return ""
	}
	return g.devName
}

// UploadWeights transfers a weight tensor to GPU device memory.
// Called once at model load time for tensors we want to accelerate.
func (g *GPUEngine) UploadWeights(name string, data []byte) error {
	if !g.ready {
		return fmt.Errorf("gpu not ready")
	}
	buf, err := VkBufAlloc(len(data))
	if err != nil {
		return fmt.Errorf("alloc %s: %w", name, err)
	}
	// Copy raw bytes
	src := data
	dst := unsafe.Slice((*byte)(buf.mapped), len(data))
	copy(dst, src)
	g.weightBufs[name] = buf
	return nil
}

// MatvecQ8_0GPU dispatches DS4 Q8_0 weight × F32 activation on GPU.
// Weight must have been pre-uploaded as raw bytes via UploadWeights.
// Returns false if kernel unavailable (caller should fall back to CPU).
func (g *GPUEngine) MatvecQ8_0GPU(out []float32, weightName string, x []float32, inDim, outDim int) bool {
	if g.gemvQ8_0 == nil {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	wBuf, ok := g.weightBufs[weightName]
	if !ok {
		return false
	}

	actSize := inDim * 4
	if g.actBuf == nil || int(g.actBuf.size) < actSize {
		if g.actBuf != nil {
			g.actBuf.Free()
		}
		var err error
		g.actBuf, err = VkBufAlloc(actSize)
		if err != nil {
			return false
		}
	}
	outSize := outDim * 4
	if g.outBuf == nil || int(g.outBuf.size) < outSize {
		if g.outBuf != nil {
			g.outBuf.Free()
		}
		var err error
		g.outBuf, err = VkBufAlloc(outSize)
		if err != nil {
			return false
		}
	}

	g.actBuf.Upload(x[:inDim])

	nBlocks := inDim / 32
	rowStride := uint32((nBlocks*34 + 3) / 4) // ceil to uint32 units
	params := struct {
		inDim, outDim, rowStride uint32
	}{uint32(inDim), uint32(outDim), rowStride}

	if err := g.gemvQ8_0.Dispatch(uint32(outDim), 1, 1,
		[]*VkBuf{g.actBuf, wBuf, g.outBuf},
		unsafe.Pointer(&params)); err != nil {
		return false
	}

	g.outBuf.Download(out[:outDim])
	return true
}

// Close releases all GPU resources.
func (g *GPUEngine) Close() {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	for name, buf := range g.weightBufs {
		buf.Free()
		delete(g.weightBufs, name)
	}
	if g.actBuf != nil {
		g.actBuf.Free()
		g.actBuf = nil
	}
	if g.outBuf != nil {
		g.outBuf.Free()
		g.outBuf = nil
	}
	// Kernels hold Vulkan pipelines/pools/fences — destroy them
	if g.gemvF32 != nil {
		g.gemvF32.Destroy()
		g.gemvF32 = nil
	}
	if g.gemvQ8_0 != nil {
		g.gemvQ8_0.Destroy()
		g.gemvQ8_0 = nil
	}
	g.ready = false
}
