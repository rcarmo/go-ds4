package ds4

import (
	"fmt"

	"github.com/rcarmo/go-ds4/ds4/gpu"
)

// InitGPU initializes the Vulkan GPU backend and uploads Q8_0 dense projection
// weights that benefit from GPU acceleration (outDim >= 4096).
func (e *Engine) InitGPU() error {
	g := gpu.GPUInit()
	if g == nil {
		return fmt.Errorf("no Vulkan GPU available")
	}
	e.GPU = g

	// Upload large Q8_0 projections (attention + shared expert + output)
	// These are the 40% of CPU time in the profile.
	uploads := []struct {
		name string
		data []byte
	}{
		// Output head (NEmbd → NVocab = 4096 → 129280)
		{"output.weight", e.Weights.Output},
	}

	// Per-layer: Q projection (attn_q_b: 1024 → 32768) and shared expert down
	for il := 0; il < NLayer; il++ {
		l := &e.Weights.Layer[il]
		prefix := fmt.Sprintf("blk.%d.", il)
		if len(l.AttnQB) > 0 {
			uploads = append(uploads, struct {
				name string
				data []byte
			}{prefix + "attn_q_b.weight", l.AttnQB})
		}
		if len(l.FfnDownShexp) > 0 {
			uploads = append(uploads, struct {
				name string
				data []byte
			}{prefix + "ffn_down_shexp.weight", l.FfnDownShexp})
		}
	}

	uploaded := 0
	var totalBytes int64
	for _, u := range uploads {
		if len(u.data) == 0 {
			continue
		}
		if err := g.UploadWeights(u.name, u.data); err != nil {
			// Non-fatal: skip this tensor, CPU fallback
			continue
		}
		uploaded++
		totalBytes += int64(len(u.data))
	}

	fmt.Printf("[gpu] Uploaded %d tensors (%.1f MB) to GPU\n", uploaded, float64(totalBytes)/(1024*1024))
	return nil
}

// GPUReady returns true if GPU acceleration is active.
func (e *Engine) GPUReady() bool {
	if e.GPU == nil {
		return false
	}
	g, ok := e.GPU.(*gpu.GPUEngine)
	return ok && g.Ready()
}

// gpuMatvecQ8_0 attempts GPU dispatch for a Q8_0 matvec.
// Returns true if dispatched on GPU, false if CPU fallback needed.
func (e *Engine) gpuMatvecQ8_0(out []float32, tensorName string, x []float32, inDim, outDim int) bool {
	if e.GPU == nil {
		return false
	}
	g, ok := e.GPU.(*gpu.GPUEngine)
	if !ok || !g.Ready() {
		return false
	}
	return g.MatvecQ8_0GPU(out, tensorName, x, inDim, outDim)
}
