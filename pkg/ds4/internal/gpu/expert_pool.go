package gpu

import (
	"fmt"
	"sync"
	"unsafe"
)

// ExpertPool manages reusable GPU buffers for streaming active expert weights.
// Each layer uses 4-6 experts; we keep a pool of device buffers sized for
// the largest expert tensor and reuse them across layers.
type ExpertPool struct {
	mu       sync.Mutex
	gateBufs []CUdeviceptr // IQ2_XXS gate weight buffers
	upBufs   []CUdeviceptr // IQ2_XXS up weight buffers
	downBufs []CUdeviceptr // Q2_K down weight buffers
	gateSize int           // bytes per expert gate tensor
	upSize   int           // bytes per expert up tensor
	downSize int           // bytes per expert down tensor
	actBuf   *Buffer       // shared f32 activation buffer
	outBufs  []*Buffer     // per-expert output buffers
	nExperts int           // max concurrent experts (4 or 6)
}

// NewExpertPool allocates GPU buffers for streaming expert weights.
func NewExpertPool(nExperts, inDim, ffnDim int) (*ExpertPool, error) {
	if !gpuOK {
		return nil, fmt.Errorf("CUDA not available")
	}
	EnsureContext()

	// IQ2_XXS: 66 bytes per 256 elements
	gateRowBytes := (inDim / 256) * 66
	gateSize := gateRowBytes * ffnDim // gate: [ffnDim, inDim]

	upRowBytes := gateRowBytes
	upSize := upRowBytes * ffnDim

	// Q2_K: 84 bytes per 256 elements
	downRowBytes := (ffnDim / 256) * 84
	downSize := downRowBytes * inDim // down: [inDim, ffnDim]

	pool := &ExpertPool{
		gateSize: gateSize,
		upSize:   upSize,
		downSize: downSize,
		nExperts: nExperts,
	}

	// Allocate per-expert weight buffers
	for i := 0; i < nExperts; i++ {
		var gp, up, dp CUdeviceptr
		if err := CuMemAllocRaw(&gp, uint64(gateSize)); err != nil {
			return nil, fmt.Errorf("alloc gate[%d]: %w", i, err)
		}
		if err := CuMemAllocRaw(&up, uint64(upSize)); err != nil {
			return nil, fmt.Errorf("alloc up[%d]: %w", i, err)
		}
		if err := CuMemAllocRaw(&dp, uint64(downSize)); err != nil {
			return nil, fmt.Errorf("alloc down[%d]: %w", i, err)
		}
		pool.gateBufs = append(pool.gateBufs, gp)
		pool.upBufs = append(pool.upBufs, up)
		pool.downBufs = append(pool.downBufs, dp)
	}

	// Shared activation buffer (max of inDim and ffnDim)
	maxDim := inDim
	if ffnDim > maxDim {
		maxDim = ffnDim
	}
	var err error
	pool.actBuf, err = Malloc(maxDim)
	if err != nil {
		return nil, err
	}

	// Per-expert output buffers
	for i := 0; i < nExperts; i++ {
		buf, err := Malloc(inDim) // expert output is [inDim]
		if err != nil {
			return nil, err
		}
		pool.outBufs = append(pool.outBufs, buf)
	}

	totalMB := float64(nExperts*(gateSize+upSize+downSize)+maxDim*4+nExperts*inDim*4) / (1024 * 1024)
	fmt.Printf("[gpu] Expert pool: %d slots, %.1f MB VRAM\n", nExperts, totalMB)
	return pool, nil
}

// UploadExpert copies one expert's weights from host (mmap) to GPU buffer slot.
func (p *ExpertPool) UploadExpert(slot int, gateData, upData, downData []byte) {
	EnsureContext()
	cuMemcpyHtoD(p.gateBufs[slot], unsafe.Pointer(&gateData[0]), uint64(len(gateData)))
	cuMemcpyHtoD(p.upBufs[slot], unsafe.Pointer(&upData[0]), uint64(len(upData)))
	cuMemcpyHtoD(p.downBufs[slot], unsafe.Pointer(&downData[0]), uint64(len(downData)))
}

// UploadActivation copies the f32 activation vector to the shared GPU buffer.
func (p *ExpertPool) UploadActivation(x []float32) {
	p.actBuf.Upload(x)
}

// GatePtr returns the device pointer for expert gate weight at slot.
func (p *ExpertPool) GatePtr(slot int) CUdeviceptr { return p.gateBufs[slot] }
func (p *ExpertPool) UpPtr(slot int) CUdeviceptr   { return p.upBufs[slot] }
func (p *ExpertPool) DownPtr(slot int) CUdeviceptr { return p.downBufs[slot] }
func (p *ExpertPool) ActBuf() *Buffer              { return p.actBuf }
func (p *ExpertPool) OutBuf(slot int) *Buffer      { return p.outBufs[slot] }

// Free releases all GPU memory in the pool.
func (p *ExpertPool) Free() {
	if p == nil {
		return
	}
	EnsureContext()
	for _, ptr := range p.gateBufs {
		CuMemFreeRaw(ptr)
	}
	for _, ptr := range p.upBufs {
		CuMemFreeRaw(ptr)
	}
	for _, ptr := range p.downBufs {
		CuMemFreeRaw(ptr)
	}
	if p.actBuf != nil {
		p.actBuf.Free()
	}
	for _, buf := range p.outBufs {
		buf.Free()
	}
}

// NExperts returns the pool capacity.
func (p *ExpertPool) NExperts() int { return p.nExperts }
