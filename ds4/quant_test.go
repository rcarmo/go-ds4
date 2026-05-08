package ds4

import (
	"math"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

func TestF16Roundtrip(t *testing.T) {
	vals := []float32{0, 1, -1, 0.5, 65504, -65504, 0.0001}
	for _, v := range vals {
		h := F32ToF16(v)
		back := F16ToF32(h)
		relErr := float64(math.Abs(float64(v-back))) / math.Max(float64(math.Abs(float64(v))), 1e-10)
		if relErr > 0.002 {
			t.Errorf("F16 roundtrip %.6f → %d → %.6f, relErr=%.6f", v, h, back, relErr)
		}
	}
}

func TestDotQ8_0F32(t *testing.T) {
	n := 64 // 2 blocks of 32
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i-32) * 0.1
	}

	// Quantize x to Q8_0 manually
	q8 := make([]byte, 2*BlockQ8_0Size)
	for b := 0; b < 2; b++ {
		off := b * BlockQ8_0Size
		xOff := b * 32
		amax := float32(0)
		for i := 0; i < 32; i++ {
			v := x[xOff+i]
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}
		scale := amax / 127.0
		*(*uint16)(unsafe.Pointer(&q8[off])) = F32ToF16(scale)
		invScale := float32(0)
		if scale != 0 {
			invScale = 1.0 / scale
		}
		for i := 0; i < 32; i++ {
			q := int8(math.Round(float64(x[xOff+i] * invScale)))
			q8[off+2+i] = byte(q)
		}
	}

	// Compute dot(q8, x)
	result := DotQ8_0F32(q8, x, n)

	// Reference: dot(x, x) — since we quantized x itself
	ref := float32(0)
	for _, v := range x {
		ref += v * v
	}

	relErr := math.Abs(float64(result-ref)) / math.Abs(float64(ref))
	t.Logf("Q8_0·F32 dot: result=%.4f ref=%.4f relErr=%.4f%%", result, ref, relErr*100)
	if relErr > 0.02 { // 2% tolerance for Q8 quantization
		t.Errorf("Q8_0 dot product error too high: %.2f%%", relErr*100)
	}
}

func TestQuantizeRowQ8K(t *testing.T) {
	x := make([]float32, QK_K)
	for i := range x {
		x[i] = float32(i-128) * 0.01
	}

	out := make([]byte, BlockQ8KSize)
	QuantizeRowQ8K(x, out)

	// Check scale
	d := *(*float32)(unsafe.Pointer(&out[0]))
	if d <= 0 {
		t.Fatalf("Q8_K scale should be positive, got %f", d)
	}

	// Check quantized values are reasonable
	maxQ := int8(0)
	for i := 0; i < QK_K; i++ {
		q := int8(out[4+i])
		if q > maxQ {
			maxQ = q
		}
	}
	if maxQ < 100 { // should be near 127 for the largest value
		t.Errorf("Q8_K max quantized value %d is too low", maxQ)
	}
	t.Logf("Q8_K: scale=%.6f maxQ=%d", d, maxQ)
}

func TestVecDotQ2KQ8K(t *testing.T) {
	// This is a smoke test with synthetic data.
	// Real validation requires actual model weights.
	n := QK_K

	// Create float32 input
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%17-8) * 0.01
	}

	// Quantize to Q8_K
	q8k := make([]byte, BlockQ8KSize)
	QuantizeRowQ8K(x, q8k)

	// Create a synthetic Q2_K block (all zeros → dot should be ~0)
	q2k := make([]byte, BlockQ2KSize)
	// Set d=1.0 in F16
	*(*uint16)(unsafe.Pointer(&q2k[80])) = F32ToF16(1.0)
	// dmin=0
	*(*uint16)(unsafe.Pointer(&q2k[82])) = 0
	// scales all zero → output should be zero
	result := VecDotQ2KQ8K(n, q2k, q8k)

	if math.Abs(float64(result)) > 1e-6 {
		t.Errorf("Q2_K·Q8_K with zero weights should be ~0, got %f", result)
	}
	t.Logf("Q2_K·Q8_K smoke test: result=%.6f", result)
}

func unsafePtr(b *byte) unsafe.Pointer {
	return unsafe.Pointer(b)
}

func TestDotI8(t *testing.T) {
	n := 256
	a := make([]int8, n)
	b := make([]int8, n)
	for i := range a {
		a[i] = int8(i%17 - 8)
		b[i] = int8(i%13 - 6)
	}
	// Reference
	ref := int32(0)
	for i := range a {
		ref += int32(a[i]) * int32(b[i])
	}
	// SIMD
	result := simd.DotI8(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), n)
	if result != ref {
		t.Errorf("DotI8: got %d, want %d", result, ref)
	}
	t.Logf("DotI8(%d): result=%d ref=%d", n, result, ref)
}

func BenchmarkDotQ8_0F32_4096(b *testing.B) {
	n := 4096
	nBlocks := n / 32
	wq8 := make([]byte, nBlocks*BlockQ8_0Size)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%17-8) * 0.01
	}
	for i := 0; i < nBlocks; i++ {
		*(*uint16)(unsafe.Pointer(&wq8[i*BlockQ8_0Size])) = F32ToF16(0.01)
		for j := 0; j < 32; j++ {
			wq8[i*BlockQ8_0Size+2+j] = byte(int8(j - 16))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotQ8_0F32(wq8, x, n)
	}
}

func BenchmarkDotF16F32_4096(b *testing.B) {
	n := 4096
	wf16 := make([]uint16, n)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%17-8) * 0.01
	}
	for i := range wf16 {
		wf16[i] = F32ToF16(x[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotF16(wf16, x)
	}
}

func BenchmarkDotI8_256(b *testing.B) {
	a := make([]int8, 256)
	bb := make([]int8, 256)
	for i := range a {
		a[i] = int8(i%17 - 8)
	}
	for i := range bb {
		bb[i] = int8(i%13 - 6)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simd.DotI8(unsafe.Pointer(&a[0]), unsafe.Pointer(&bb[0]), 256)
	}
}
