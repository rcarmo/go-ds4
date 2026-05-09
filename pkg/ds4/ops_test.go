package ds4

import (
	"math"
	"testing"
)

func TestSoftmax(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	softmax(x)
	sum := float32(0)
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
	// Values should be monotonically increasing
	for i := 1; i < len(x); i++ {
		if x[i] < x[i-1] {
			t.Errorf("softmax not monotonic: x[%d]=%f < x[%d]=%f", i, x[i], i-1, x[i-1])
		}
	}
	t.Logf("softmax([1,2,3,4]) = %v", x)
}

func TestSigmoid(t *testing.T) {
	if s := sigmoid(0); math.Abs(float64(s-0.5)) > 1e-6 {
		t.Errorf("sigmoid(0) = %f, want 0.5", s)
	}
	if s := sigmoid(100); math.Abs(float64(s-1.0)) > 1e-4 {
		t.Errorf("sigmoid(100) = %f, want ~1.0", s)
	}
	if s := sigmoid(-100); math.Abs(float64(s)) > 1e-4 {
		t.Errorf("sigmoid(-100) = %f, want ~0.0", s)
	}
}

func TestRoPEYaRNTailInplace(t *testing.T) {
	headDim := 512
	nRot := 64
	nHead := 2
	q := make([]float32, nHead*headDim)
	// Fill with ones in the tail region
	for h := 0; h < nHead; h++ {
		for i := headDim - nRot; i < headDim; i++ {
			q[h*headDim+i] = 1.0
		}
	}

	// Apply RoPE at position 0 — should be identity (cos(0)=1, sin(0)=0)
	qCopy := make([]float32, len(q))
	copy(qCopy, q)
	ropeYaRNTailInplace(q, 0, nHead, headDim, nRot, RoPEFreqBase, 1.0, false)

	// At pos=0, rotation is identity
	maxDiff := float32(0)
	for i := range q {
		d := q[i] - qCopy[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-5 {
		t.Errorf("RoPE at pos=0 should be identity, maxDiff=%f", maxDiff)
	}

	// Apply at pos=1, then inverse at pos=1 → should recover original
	copy(q, qCopy)
	ropeYaRNTailInplace(q, 1, nHead, headDim, nRot, RoPEFreqBase, 1.0, false)
	ropeYaRNTailInplace(q, 1, nHead, headDim, nRot, RoPEFreqBase, 1.0, true)

	maxDiff = 0
	for i := range q {
		d := q[i] - qCopy[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-4 {
		t.Errorf("RoPE forward+inverse should recover original, maxDiff=%f", maxDiff)
	}
	t.Logf("RoPE round-trip max diff: %e", maxDiff)
}

func TestFP8E4M3FNTable(t *testing.T) {
	// e4m3fnTable[0] should be 0
	if e4m3fnTable[0] != 0 {
		t.Errorf("e4m3fn[0] = %f, want 0", e4m3fnTable[0])
	}
	// e4m3fnTable[0x3F] = 1.0 * (1+7/8) * 2^(7-7) = 1.875
	// Actually let me check: exp=7, mant=7 → (1+7/8) * 2^(7-7) = 1.875
	// But index 0x3F = 0b00111111 → exp=7 (bits 6-3), mant=7 (bits 2-0)
	// Wait: bit layout is |S|EEEE|MMM| = |0|0111|111| = 0x3F
	v := e4m3fnTable[0x3F]
	expected := float32(1.875) // (1+7/8) * 2^(7-7)
	if math.Abs(float64(v-expected)) > 1e-6 {
		t.Errorf("e4m3fn[0x3F] = %f, want %f", v, expected)
	}

	// 0x78 = |0|1111|000| → exp=15, mant=0 → (1+0) * 2^(15-7) = 256
	v = e4m3fnTable[0x78]
	if math.Abs(float64(v-256)) > 1e-4 {
		t.Errorf("e4m3fn[0x78] = %f, want 256", v)
	}

	// Max non-NaN: 0x7E = |0|1111|110| → exp=15, mant=6 → (1+6/8)*2^8 = 448
	v = e4m3fnTable[0x7E]
	if math.Abs(float64(v-448)) > 1e-4 {
		t.Errorf("e4m3fn[0x7E] = %f, want 448", v)
	}
	t.Logf("E4M3FN max = %f (0x7E), NaN at 0x7F = %f", e4m3fnTable[0x7E], e4m3fnTable[0x7F])
}

func TestParallelFor(t *testing.T) {
	n := 1000
	results := make([]int, n)
	parallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			results[i] = i * 2
		}
	})
	for i := 0; i < n; i++ {
		if results[i] != i*2 {
			t.Fatalf("results[%d] = %d, want %d", i, results[i], i*2)
		}
	}
}
