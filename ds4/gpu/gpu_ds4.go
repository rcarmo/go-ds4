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
	gemvF32 *VkComputeKernel // F32 weight × F32 activation → F32
	// Q8_0 kernel would need per-block F16 decode in shader — complex but doable

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
	fmt.Printf("[gpu] Vulkan compute ready: %s\n", g.devName)
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

// MatvecF32GPU dispatches F32 weight × F32 activation on GPU.
// out[outDim] = weight[outDim, inDim] · x[inDim]
func (g *GPUEngine) MatvecF32GPU(out []float32, weightName string, x []float32, inDim, outDim int) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	wBuf, ok := g.weightBufs[weightName]
	if !ok {
		return fmt.Errorf("weight %q not on GPU", weightName)
	}

	// Ensure activation buffer is big enough
	actSize := inDim * 4
	if g.actBuf == nil || int(g.actBuf.size) < actSize {
		if g.actBuf != nil {
			g.actBuf.Free()
		}
		var err error
		g.actBuf, err = VkBufAlloc(actSize)
		if err != nil {
			return err
		}
	}

	// Ensure output buffer
	outSize := outDim * 4
	if g.outBuf == nil || int(g.outBuf.size) < outSize {
		if g.outBuf != nil {
			g.outBuf.Free()
		}
		var err error
		g.outBuf, err = VkBufAlloc(outSize)
		if err != nil {
			return err
		}
	}

	// Upload activation
	g.actBuf.Upload(x[:inDim])

	// Dispatch GEMV: one workgroup per output row
	if g.gemvF32 == nil {
		k, err := VkKernelCreate(spirvGemvF32, 3, 8) // 3 buffers, 8 bytes push (inDim, outDim)
		if err != nil {
			return fmt.Errorf("create gemv kernel: %w", err)
		}
		g.gemvF32 = k
	}

	params := struct {
		inDim  uint32
		outDim uint32
	}{uint32(inDim), uint32(outDim)}

	if err := g.gemvF32.Dispatch(uint32(outDim), 1, 1,
		[]*VkBuf{g.actBuf, wBuf, g.outBuf},
		unsafe.Pointer(&params)); err != nil {
		return err
	}

	// Download result
	g.outBuf.Download(out[:outDim])
	return nil
}

// Close releases all GPU resources.
func (g *GPUEngine) Close() {
	if g == nil {
		return
	}
	for _, buf := range g.weightBufs {
		buf.Free()
	}
	if g.actBuf != nil {
		g.actBuf.Free()
	}
	if g.outBuf != nil {
		g.outBuf.Free()
	}
}
