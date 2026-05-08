package gpu

import (
	"fmt"
	"unsafe"
)

// FusedQ8Buf holds concatenated Q8_0 weight buffers for fused matmul dispatch.
// Multiple small projections sharing the same activation are concatenated
// into one weight buffer and dispatched as one kernel.
type FusedQ8Buf struct {
	Ptr    CUdeviceptr // device pointer to concatenated weights
	OutDim int         // total output dimension (sum of all fused)
	Splits []int       // output dimension of each sub-projection
	OutBuf *Buffer     // [OutDim] output buffer
	ActBuf *Buffer     // [InDim] shared activation buffer
	InDim  int
}

// NewFusedQ8Buf creates a fused weight buffer from multiple Q8_0 weight tensors.
// All tensors must have the same input dimension.
func NewFusedQ8Buf(inDim int, weights [][]byte, outDims []int) (*FusedQ8Buf, error) {
	if !gpuOK {
		return nil, fmt.Errorf("CUDA not available")
	}
	EnsureContext()

	totalOutDim := 0
	for _, d := range outDims {
		totalOutDim += d
	}

	rowBytes := (inDim / 32) * 34 // Q8_0 block size
	totalBytes := totalOutDim * rowBytes

	var ptr CUdeviceptr
	if err := CuMemAllocRaw(&ptr, uint64(totalBytes)); err != nil {
		return nil, err
	}

	// Copy each weight tensor contiguously
	offset := 0
	for i, w := range weights {
		size := outDims[i] * rowBytes
		if len(w) < size {
			continue
		}
		cuMemcpyHtoD(CUdeviceptr(uint64(ptr)+uint64(offset)), unsafe.Pointer(&w[0]), uint64(size))
		offset += size
	}

	outBuf, err := Malloc(totalOutDim)
	if err != nil {
		CuMemFreeRaw(ptr)
		return nil, err
	}
	actBuf, err := Malloc(inDim)
	if err != nil {
		CuMemFreeRaw(ptr)
		outBuf.Free()
		return nil, err
	}

	return &FusedQ8Buf{
		Ptr:    ptr,
		OutDim: totalOutDim,
		Splits: outDims,
		OutBuf: outBuf,
		ActBuf: actBuf,
		InDim:  inDim,
	}, nil
}

// Dispatch runs the fused Q8_0 GEMV and splits output into multiple result slices.
func (f *FusedQ8Buf) Dispatch(x []float32, results [][]float32, stream CUstream) error {
	f.ActBuf.UploadAsync(x, stream)

	rowBytes := (f.InDim / 32) * 34
	nBlocks := uint32(f.InDim / 32)
	outDim := uint32(f.OutDim)
	rowBytesU := uint32(rowBytes)
	actPtr := f.ActBuf.Ptr
	outPtr := f.OutBuf.Ptr
	wtPtr := f.Ptr

	args := []unsafe.Pointer{
		unsafe.Pointer(&actPtr),
		unsafe.Pointer(&wtPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&nBlocks),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&rowBytesU),
	}

	if err := LaunchKernelStream(cudaGemvQ8_0, uint32(f.OutDim), 1, 1, 256, 1, 1, 256*4, stream, args...); err != nil {
		return err
	}

	StreamSync(stream)

	// Download and split
	allOut := make([]float32, f.OutDim)
	f.OutBuf.Download(allOut)

	offset := 0
	for i, dim := range f.Splits {
		if i < len(results) {
			copy(results[i], allOut[offset:offset+dim])
		}
		offset += dim
	}
	return nil
}

// Free releases GPU memory.
func (f *FusedQ8Buf) Free() {
	if f == nil {
		return
	}
	EnsureContext()
	CuMemFreeRaw(f.Ptr)
	if f.OutBuf != nil {
		f.OutBuf.Free()
	}
	if f.ActBuf != nil {
		f.ActBuf.Free()
	}
}

// FusedLayerBufs holds fused weight buffers for one transformer layer.
type FusedLayerBufs struct {
	AttnQAKV *FusedQ8Buf // attn_q_a [1024] + attn_kv [512] = [1536]
}

func NewFusedLayerBufs(attnQA, attnKV []byte, inDim, qaOutDim, kvOutDim int) (*FusedLayerBufs, error) {
	fused, err := NewFusedQ8Buf(inDim, [][]byte{attnQA, attnKV}, []int{qaOutDim, kvOutDim})
	if err != nil {
		return nil, err
	}
	return &FusedLayerBufs{AttnQAKV: fused}, nil
}

func (f *FusedLayerBufs) Free() {
	if f == nil {
		return
	}
	if f.AttnQAKV != nil {
		f.AttnQAKV.Free()
	}
}
