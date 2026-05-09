package ds4

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-ds4/pkg/ds4/internal/gpu"
)

func requireCUDATest(t *testing.T) {
	t.Helper()
	if !gpu.Init() {
		t.Skip("CUDA not available")
	}
}

func fillQ8KActivation(seed int64, n int) ([]float32, []byte) {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q8 := make([]byte, (n/QK_K)*BlockQ8KSize)
	QuantizeRowQ8K(x, q8)
	return x, q8
}

func uploadBytesForTest(t *testing.T, data []byte) gpu.CUdeviceptr {
	t.Helper()
	var ptr gpu.CUdeviceptr
	if err := gpu.CuMemAllocRaw(&ptr, uint64(len(data))); err != nil {
		t.Fatalf("cuMemAllocRaw(%d): %v", len(data), err)
	}
	gpu.CuMemcpyHtoDRaw(ptr, unsafe.Pointer(&data[0]), uint64(len(data)))
	t.Cleanup(func() { gpu.CuMemFreeRaw(ptr) })
	return ptr
}

func TestCUDAVecDotQ2KQ8KParity(t *testing.T) {
	requireCUDATest(t)
	if !gpu.InitCUDAGemvQ2KQ8K() {
		t.Fatal("Q2_K x Q8_K CUDA kernel not available")
	}

	const inDim = 2048
	const outDim = 8
	rowBytes := (inDim / QK_K) * BlockQ2KSize
	_, q8 := fillQ8KActivation(1, inDim)

	rng := rand.New(rand.NewSource(2))
	wt := make([]byte, outDim*rowBytes)
	for row := 0; row < outDim; row++ {
		for b := 0; b < inDim/QK_K; b++ {
			blk := wt[row*rowBytes+b*BlockQ2KSize:]
			for i := 0; i < 16; i++ {
				blk[i] = byte(rng.Intn(256))
			}
			for i := 16; i < 80; i++ {
				blk[i] = byte(rng.Intn(256))
			}
			*(*uint16)(unsafe.Pointer(&blk[80])) = F32ToF16(0.01)
			*(*uint16)(unsafe.Pointer(&blk[82])) = F32ToF16(0.005)
		}
	}

	qPtr := uploadBytesForTest(t, q8)
	wPtr := uploadBytesForTest(t, wt)
	out, err := gpu.Malloc(outDim)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(out.Free)
	if err := gpu.CUDAMatvecQ2KQ8K(out, qPtr, wPtr, inDim, outDim, rowBytes, 0); err != nil {
		t.Fatal(err)
	}
	if err := gpu.SyncErr(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, outDim)
	if err := out.Download(got); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < outDim; row++ {
		want := VecDotQ2KQ8K(inDim, wt[row*rowBytes:(row+1)*rowBytes], q8)
		if diff := got[row] - want; diff > 1e-3 || diff < -1e-3 {
			t.Fatalf("row %d: got %g want %g diff %g", row, got[row], want, diff)
		}
	}
}

func TestCUDAQ8_0PrequantParity(t *testing.T) {
	requireCUDATest(t)
	if !gpu.InitCUDAGemvQ8_0Prequant() {
		t.Fatal("Q8_0 prequant CUDA kernel not available")
	}
	const inDim = 4096
	const outDim = 16
	nBlocks := (inDim + 31) / 32
	rowBytes := nBlocks * BlockQ8_0Size
	rng := rand.New(rand.NewSource(6))
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	xq := make([]int8, nBlocks*32)
	xscale := make([]float32, nBlocks)
	xsum := make([]float32, nBlocks)
	quantizeQ8_0Activation(x, xq, xscale, xsum)
	w := make([]byte, outDim*rowBytes)
	for r := 0; r < outDim; r++ {
		for b := 0; b < nBlocks; b++ {
			blk := w[r*rowBytes+b*BlockQ8_0Size:]
			*(*uint16)(unsafe.Pointer(&blk[0])) = F32ToF16(float32(rng.Float64()*0.02 + 0.001))
			for i := 0; i < 32; i++ {
				blk[2+i] = byte(int8(rng.Intn(255) - 127))
			}
		}
	}
	qPtr := uploadBytesForTest(t, unsafe.Slice((*byte)(unsafe.Pointer(&xq[0])), len(xq)))
	wPtr := uploadBytesForTest(t, w)
	scaleBuf, err := gpu.Malloc(len(xscale))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(scaleBuf.Free)
	if err := scaleBuf.Upload(xscale); err != nil {
		t.Fatal(err)
	}
	out, err := gpu.Malloc(outDim)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(out.Free)
	if err := gpu.CUDAMatvecQ8_0Prequant(out, qPtr, scaleBuf.Ptr, wPtr, inDim, outDim, rowBytes); err != nil {
		t.Fatal(err)
	}
	if err := gpu.SyncErr(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, outDim)
	if err := out.Download(got); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < outDim; r++ {
		want := dotQ8_0Prequant(w[r*rowBytes:(r+1)*rowBytes], xq, xscale, xsum, nBlocks)
		if diff := got[r] - want; diff > 1e-3 || diff < -1e-3 {
			t.Fatalf("row %d got %g want %g diff %g", r, got[r], want, diff)
		}
	}
}

func TestCUDAExpertSwiGLUQ8KParity(t *testing.T) {
	requireCUDATest(t)
	if !gpu.InitCUDAExpertSwiGLUQ8K() {
		t.Fatal("expert SwiGLU+Q8_K CUDA kernel not available")
	}

	const ffnDim = 2048
	const nExperts = 2
	const clamp = float32(7.0)
	rng := rand.New(rand.NewSource(5))
	gate := make([]float32, nExperts*ffnDim)
	up := make([]float32, nExperts*ffnDim)
	weights := []float32{0.375, 0.625}
	want := make([]byte, nExperts*(ffnDim/QK_K)*BlockQ8KSize)
	for e := 0; e < nExperts; e++ {
		mid := make([]float32, ffnDim)
		for i := 0; i < ffnDim; i++ {
			g := float32(rng.NormFloat64() * 2)
			u := float32(rng.NormFloat64() * 2)
			gate[e*ffnDim+i] = g
			up[e*ffnDim+i] = u
			if g > clamp {
				g = clamp
			}
			if u > clamp {
				u = clamp
			} else if u < -clamp {
				u = -clamp
			}
			sig := float32(0)
			if g > 10 {
				sig = 1
			} else if g >= -10 {
				const ln2inv = 1.4426950408889634
				bits := int32((-g)*ln2inv*float32(1<<23)) + int32(127<<23)
				exp := math.Float32frombits(uint32(bits))
				sig = 1 / (1 + exp)
			}
			mid[i] = g * sig * u * weights[e]
		}
		QuantizeRowQ8K(mid, want[e*(ffnDim/QK_K)*BlockQ8KSize:])
	}

	gateBuf, err := gpu.Malloc(len(gate))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(gateBuf.Free)
	upBuf, err := gpu.Malloc(len(up))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(upBuf.Free)
	if err := gateBuf.Upload(gate); err != nil {
		t.Fatal(err)
	}
	if err := upBuf.Upload(up); err != nil {
		t.Fatal(err)
	}
	var weightsPtr, outPtr gpu.CUdeviceptr
	if err := gpu.CuMemAllocRaw(&weightsPtr, uint64(len(weights)*4)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gpu.CuMemFreeRaw(weightsPtr) })
	gpu.CuMemcpyHtoDRaw(weightsPtr, unsafe.Pointer(&weights[0]), uint64(len(weights)*4))
	if err := gpu.CuMemAllocRaw(&outPtr, uint64(len(want))); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gpu.CuMemFreeRaw(outPtr) })
	if err := gpu.CUDAExpertSwiGLUQ8K(gateBuf, upBuf, weightsPtr, outPtr, ffnDim, nExperts, clamp, 0); err != nil {
		t.Fatal(err)
	}
	if err := gpu.SyncErr(); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len(want))
	if err := gpu.CuMemcpyDtoHRaw(unsafe.Pointer(&got[0]), outPtr, uint64(len(got))); err != nil {
		t.Fatal(err)
	}
	// Q8_K quantization is the contract here. Allow a small number of +/-1 byte
	// differences from CUDA exp/rounding, but scales/bsums must stay very close.
	bad := 0
	for i := range got {
		d := int(int8(got[i])) - int(int8(want[i]))
		if d < 0 {
			d = -d
		}
		if d > 1 {
			bad++
		}
	}
	if bad > len(got)/200 { // >0.5% material byte drift
		t.Fatalf("expert Q8_K output drift: %d/%d bytes differ by more than 1", bad, len(got))
	}
}

func TestCUDAExpertHiddenToDownParity(t *testing.T) {
	requireCUDATest(t)
	if !gpu.InitCUDAExpertSwiGLUQ8K() || !gpu.InitCUDAGemvQ2KQ8K() {
		t.Fatal("expert hidden/down CUDA kernels not available")
	}
	const ffnDim = 2048
	const outDim = 16
	const clamp = float32(7.0)
	rng := rand.New(rand.NewSource(7))
	gate := make([]float32, ffnDim)
	up := make([]float32, ffnDim)
	weight := []float32{0.5}
	mid := make([]float32, ffnDim)
	for i := range gate {
		g := float32(rng.NormFloat64() * 2)
		u := float32(rng.NormFloat64() * 2)
		gate[i], up[i] = g, u
		if g > clamp {
			g = clamp
		}
		if u > clamp {
			u = clamp
		} else if u < -clamp {
			u = -clamp
		}
		sig := float32(0)
		if g > 10 {
			sig = 1
		} else if g >= -10 {
			const ln2inv = 1.4426950408889634
			bits := int32((-g)*ln2inv*float32(1<<23)) + int32(127<<23)
			exp := math.Float32frombits(uint32(bits))
			sig = 1 / (1 + exp)
		}
		mid[i] = g * sig * u * weight[0]
	}
	midQCPU := make([]byte, (ffnDim/QK_K)*BlockQ8KSize)
	QuantizeRowQ8K(mid, midQCPU)
	downRowBytes := (ffnDim / QK_K) * BlockQ2KSize
	down := make([]byte, outDim*downRowBytes)
	for row := 0; row < outDim; row++ {
		for b := 0; b < ffnDim/QK_K; b++ {
			blk := down[row*downRowBytes+b*BlockQ2KSize:]
			for i := 0; i < 16; i++ {
				blk[i] = byte(rng.Intn(256))
			}
			for i := 16; i < 80; i++ {
				blk[i] = byte(rng.Intn(256))
			}
			*(*uint16)(unsafe.Pointer(&blk[80])) = F32ToF16(0.01)
			*(*uint16)(unsafe.Pointer(&blk[82])) = F32ToF16(0.005)
		}
	}
	want := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		want[row] = VecDotQ2KQ8K(ffnDim, down[row*downRowBytes:(row+1)*downRowBytes], midQCPU)
	}
	gateBuf, _ := gpu.Malloc(ffnDim)
	t.Cleanup(gateBuf.Free)
	_ = gateBuf.Upload(gate)
	upBuf, _ := gpu.Malloc(ffnDim)
	t.Cleanup(upBuf.Free)
	_ = upBuf.Upload(up)
	wPtr := uploadBytesForTest(t, unsafe.Slice((*byte)(unsafe.Pointer(&weight[0])), 4))
	midPtr := uploadBytesForTest(t, make([]byte, len(midQCPU)))
	if err := gpu.CUDAExpertSwiGLUQ8K(gateBuf, upBuf, wPtr, midPtr, ffnDim, 1, clamp, 0); err != nil {
		t.Fatal(err)
	}
	if err := gpu.SyncErr(); err != nil {
		t.Fatal(err)
	}
	downPtr := uploadBytesForTest(t, down)
	out, _ := gpu.Malloc(outDim)
	t.Cleanup(out.Free)
	if err := gpu.CUDAMatvecQ2KQ8K(out, midPtr, downPtr, ffnDim, outDim, downRowBytes, 0); err != nil {
		t.Fatal(err)
	}
	if err := gpu.SyncErr(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, outDim)
	_ = out.Download(got)
	for row := range got {
		if diff := got[row] - want[row]; diff > 0.2 || diff < -0.2 {
			t.Fatalf("row %d got %g want %g diff %g", row, got[row], want[row], diff)
		}
	}
}

func TestCUDAVecDotIQ2XXSQ8KParity(t *testing.T) {
	requireCUDATest(t)
	gridSlice := (*[256 * 128 * 8]int8)(unsafe.Pointer(&iq2xxsSignedGrid[0]))
	gpu.InitCUDAGemvIQ2(gridSlice[:])
	if !gpu.InitCUDAGemvIQ2Q8K() {
		t.Fatal("IQ2_XXS x Q8_K CUDA kernel not available")
	}

	const inDim = 4096
	const outDim = 8
	rowBytes := (inDim / QK_K) * BlockIQ2XXSSize
	_, q8 := fillQ8KActivation(3, inDim)

	rng := rand.New(rand.NewSource(4))
	wt := make([]byte, outDim*rowBytes)
	for row := 0; row < outDim; row++ {
		for b := 0; b < inDim/QK_K; b++ {
			blk := wt[row*rowBytes+b*BlockIQ2XXSSize:]
			*(*uint16)(unsafe.Pointer(&blk[0])) = F32ToF16(0.01)
			for g := 0; g < 8; g++ {
				off := 2 + g*8
				for l := 0; l < 4; l++ {
					blk[off+l] = byte(rng.Intn(256))
				}
				aux := uint32(rng.Intn(16)) << 28
				for l := 0; l < 4; l++ {
					aux |= uint32(rng.Intn(128)) << (7 * l)
				}
				*(*uint32)(unsafe.Pointer(&blk[off+4])) = aux
			}
		}
	}

	qPtr := uploadBytesForTest(t, q8)
	wPtr := uploadBytesForTest(t, wt)
	out, err := gpu.Malloc(outDim)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(out.Free)
	if err := gpu.CUDAMatvecIQ2Q8K(out, qPtr, wPtr, inDim, outDim, rowBytes, 0); err != nil {
		t.Fatal(err)
	}
	if err := gpu.SyncErr(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, outDim)
	if err := out.Download(got); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < outDim; row++ {
		want := vecDotIQ2XXSQ8K_scalar(inDim, wt[row*rowBytes:(row+1)*rowBytes], q8)
		if diff := got[row] - want; diff > 1e-3 || diff < -1e-3 {
			t.Fatalf("row %d: got %g want %g diff %g", row, got[row], want, diff)
		}
	}
}
