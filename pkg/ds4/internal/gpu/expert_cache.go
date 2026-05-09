package gpu

import (
	"fmt"
	"unsafe"
)

// ExpertCache holds permanently-resident expert weights in VRAM.
// Pre-loaded at init time based on frequency analysis or fixed top-N.
// No per-token PCIe transfer needed for cached experts.
type ExpertCache struct {
	// Per layer: map[expertIdx] → device pointers
	layers [43]layerExpertCache

	nCached   int // experts cached per layer
	gateSize  int // bytes per gate tensor
	upSize    int // bytes per up tensor
	downSize  int // bytes per down tensor
	totalVRAM int64
}

type layerExpertCache struct {
	cached map[int]*cachedExpert // expertIdx → GPU pointers
	actBuf *Buffer               // shared activation buffer for this layer
	outBuf *Buffer               // shared output buffer
	midBuf *Buffer               // intermediate buffer (gate/up result)
}

type cachedExpert struct {
	gatePtr CUdeviceptr
	upPtr   CUdeviceptr
	downPtr CUdeviceptr
}

// NewExpertCache allocates permanent VRAM for top-N experts per layer.
func NewExpertCache(nPerLayer int) (*ExpertCache, error) {
	if !gpuOK {
		return nil, fmt.Errorf("CUDA not available")
	}
	EnsureContext()

	NEmbd := 4096
	NFFExp := 2048
	NLayer := 43

	gateSize := (NEmbd / 256) * 66 * NFFExp // IQ2_XXS [NFFExp, NEmbd]
	upSize := gateSize                      // same format
	downSize := (NFFExp / 256) * 84 * NEmbd // Q2_K [NEmbd, NFFExp]

	ec := &ExpertCache{
		nCached:  nPerLayer,
		gateSize: gateSize,
		upSize:   upSize,
		downSize: downSize,
	}

	for il := 0; il < NLayer; il++ {
		ec.layers[il].cached = make(map[int]*cachedExpert, nPerLayer)

		// Shared buffers per layer (reused across experts)
		var err error
		ec.layers[il].actBuf, err = Malloc(NEmbd)
		if err != nil {
			return nil, fmt.Errorf("layer %d actBuf: %w", il, err)
		}
		ec.layers[il].outBuf, err = Malloc(NEmbd)
		if err != nil {
			return nil, fmt.Errorf("layer %d outBuf: %w", il, err)
		}
		ec.layers[il].midBuf, err = Malloc(NFFExp)
		if err != nil {
			return nil, fmt.Errorf("layer %d midBuf: %w", il, err)
		}
	}

	return ec, nil
}

// LoadExpert uploads one expert's weights to permanent VRAM cache.
func (ec *ExpertCache) LoadExpert(il, expertIdx int, gateData, upData, downData []byte) error {
	EnsureContext()

	ce := &cachedExpert{}
	var err error

	if err = CuMemAllocRaw(&ce.gatePtr, uint64(ec.gateSize)); err != nil {
		return err
	}
	if err = CuMemAllocRaw(&ce.upPtr, uint64(ec.upSize)); err != nil {
		CuMemFreeRaw(ce.gatePtr)
		return err
	}
	if err = CuMemAllocRaw(&ce.downPtr, uint64(ec.downSize)); err != nil {
		CuMemFreeRaw(ce.gatePtr)
		CuMemFreeRaw(ce.upPtr)
		return err
	}

	cuMemcpyHtoD(ce.gatePtr, unsafe.Pointer(&gateData[0]), uint64(len(gateData)))
	cuMemcpyHtoD(ce.upPtr, unsafe.Pointer(&upData[0]), uint64(len(upData)))
	cuMemcpyHtoD(ce.downPtr, unsafe.Pointer(&downData[0]), uint64(len(downData)))

	ec.layers[il].cached[expertIdx] = ce
	ec.totalVRAM += int64(ec.gateSize + ec.upSize + ec.downSize)
	return nil
}

// IsCached returns true if the expert is in VRAM.
func (ec *ExpertCache) IsCached(il, expertIdx int) bool {
	_, ok := ec.layers[il].cached[expertIdx]
	return ok
}

// Get returns the cached expert pointers.
func (ec *ExpertCache) Get(il, expertIdx int) (gate, up, down CUdeviceptr, ok bool) {
	ce, exists := ec.layers[il].cached[expertIdx]
	if !exists {
		return 0, 0, 0, false
	}
	return ce.gatePtr, ce.upPtr, ce.downPtr, true
}

// ActBuf returns the shared activation buffer for a layer.
func (ec *ExpertCache) ActBuf(il int) *Buffer { return ec.layers[il].actBuf }

// OutBuf returns the shared output buffer for a layer.
func (ec *ExpertCache) OutBuf(il int) *Buffer { return ec.layers[il].outBuf }

// MidBuf returns the shared intermediate buffer for a layer.
func (ec *ExpertCache) MidBuf(il int) *Buffer { return ec.layers[il].midBuf }

// TotalVRAM returns bytes of expert weights cached.
func (ec *ExpertCache) TotalVRAM() int64 { return ec.totalVRAM }

// NCached returns experts cached per layer.
func (ec *ExpertCache) NCached() int { return ec.nCached }

// Free releases all VRAM.
func (ec *ExpertCache) Free() {
	if ec == nil {
		return
	}
	EnsureContext()
	for il := range ec.layers {
		for _, ce := range ec.layers[il].cached {
			CuMemFreeRaw(ce.gatePtr)
			CuMemFreeRaw(ce.upPtr)
			CuMemFreeRaw(ce.downPtr)
		}
		if ec.layers[il].actBuf != nil {
			ec.layers[il].actBuf.Free()
		}
		if ec.layers[il].outBuf != nil {
			ec.layers[il].outBuf.Free()
		}
		if ec.layers[il].midBuf != nil {
			ec.layers[il].midBuf.Free()
		}
	}
}

// BatchedExpertBufs holds contiguous GPU memory for batched expert dispatch.
type BatchedExpertBufs struct {
	GateBuf  CUdeviceptr // contiguous gate weights for N experts
	UpBuf    CUdeviceptr // contiguous up weights
	DownBuf  CUdeviceptr // contiguous down weights
	GateOut  *Buffer     // [N*NFFExp] gate output
	UpOut    *Buffer     // [N*NFFExp] up output
	DownOut  *Buffer     // [N*NEmbd] down output
	ActBuf   *Buffer     // [NEmbd] shared activation
	nExperts int
}

func NewBatchedExpertBufs(nExperts, inDim, ffnDim int) (*BatchedExpertBufs, error) {
	EnsureContext()
	gateSize := (inDim / 256) * 66 * ffnDim
	upSize := gateSize
	downSize := (ffnDim / 256) * 84 * inDim

	b := &BatchedExpertBufs{nExperts: nExperts}
	var err error
	if err = CuMemAllocRaw(&b.GateBuf, uint64(nExperts*gateSize)); err != nil {
		return nil, err
	}
	if err = CuMemAllocRaw(&b.UpBuf, uint64(nExperts*upSize)); err != nil {
		return nil, err
	}
	if err = CuMemAllocRaw(&b.DownBuf, uint64(nExperts*downSize)); err != nil {
		return nil, err
	}
	b.GateOut, err = Malloc(nExperts * ffnDim)
	if err != nil {
		return nil, err
	}
	b.UpOut, err = Malloc(nExperts * ffnDim)
	if err != nil {
		return nil, err
	}
	b.DownOut, err = Malloc(nExperts * inDim)
	if err != nil {
		return nil, err
	}
	b.ActBuf, err = Malloc(inDim)
	if err != nil {
		return nil, err
	}
	return b, nil
}

// CopyExpertWeights copies cached expert weights contiguously for batched dispatch.
func (b *BatchedExpertBufs) CopyExpertWeights(slot int, gatePtr, upPtr, downPtr CUdeviceptr, gateSize, upSize, downSize int) {
	EnsureContext()
	cuMemcpyDtoD(CUdeviceptr(uint64(b.GateBuf)+uint64(slot*gateSize)), gatePtr, uint64(gateSize))
	cuMemcpyDtoD(CUdeviceptr(uint64(b.UpBuf)+uint64(slot*upSize)), upPtr, uint64(upSize))
	cuMemcpyDtoD(CUdeviceptr(uint64(b.DownBuf)+uint64(slot*downSize)), downPtr, uint64(downSize))
}

func (b *BatchedExpertBufs) Free() {
	if b == nil {
		return
	}
	EnsureContext()
	CuMemFreeRaw(b.GateBuf)
	CuMemFreeRaw(b.UpBuf)
	CuMemFreeRaw(b.DownBuf)
	if b.GateOut != nil {
		b.GateOut.Free()
	}
	if b.UpOut != nil {
		b.UpOut.Free()
	}
	if b.DownOut != nil {
		b.DownOut.Free()
	}
	if b.ActBuf != nil {
		b.ActBuf.Free()
	}
}
