package ds4

import (
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
