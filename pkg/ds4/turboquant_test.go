package ds4

import (
	"math"
	"testing"
)

func TestFWHTRoundtrip(t *testing.T) {
	for _, dim := range []int{8, 16, 64, 512} {
		x := make([]float32, dim)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
		}
		orig := make([]float32, dim)
		copy(orig, x)
		Fwht(x)
		Fwht(x)
		for i := range x {
			x[i] /= float32(dim)
		}
		maxErr := float32(0)
		for i := range x {
			e := x[i] - orig[i]
			if e < 0 {
				e = -e
			}
			if e > maxErr {
				maxErr = e
			}
		}
		if maxErr > 1e-5 {
			t.Errorf("dim=%d: FWHT roundtrip error %e", dim, maxErr)
		}
	}
}

func TestTurboQuantCompression(t *testing.T) {
	dim := 448
	for _, bits := range []int{2, 3, 4} {
		cfg := NewTurboQuantConfig(dim, bits)
		t.Logf("%d-bit: %d F32 bytes → %d compressed (%.1fx)",
			bits, dim*4, cfg.CompressedSize(), float64(dim*4)/float64(cfg.CompressedSize()))
	}
}

func BenchmarkFWHT512(b *testing.B) {
	x := make([]float32, 512)
	for i := range x {
		x[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Fwht(x)
	}
}

func TestTurboQuantRoundtrip(t *testing.T) {
	for _, bits := range []int{2, 3, 4} {
		dim := 448
		cfg := NewTurboQuantConfig(dim, bits)
		x := make([]float32, dim)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
		}

		comp := make([]byte, cfg.CompressedSize())
		cfg.Encode(x, comp)
		out := make([]float32, dim)
		cfg.Decode(comp, out)

		var mse, normX float64
		for i := range x {
			d := float64(x[i] - out[i])
			mse += d * d
			normX += float64(x[i]) * float64(x[i])
		}
		rmse := math.Sqrt(mse / float64(dim))
		relErr := rmse / math.Sqrt(normX/float64(dim))
		t.Logf("%d-bit: RMSE=%.6f relErr=%.2f%% compression=%.1fx",
			bits, rmse, relErr*100, float64(dim*4)/float64(cfg.CompressedSize()))
	}
}

func TestTurboQuantPow2(t *testing.T) {
	// Test with power-of-2 dim (no padding needed)
	for _, bits := range []int{3, 4, 8} {
		dim := 512 // NHeadDim is already pow2!
		cfg := NewTurboQuantConfig(dim, bits)
		x := make([]float32, dim)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
		}
		comp := make([]byte, cfg.CompressedSize())
		cfg.Encode(x, comp)
		out := make([]float32, dim)
		cfg.Decode(comp, out)
		var mse, normX float64
		for i := range x {
			d := float64(x[i] - out[i])
			mse += d * d
			normX += float64(x[i]) * float64(x[i])
		}
		rmse := math.Sqrt(mse / float64(dim))
		relErr := rmse / math.Sqrt(normX/float64(dim))
		t.Logf("%d-bit dim=%d: RMSE=%.6f relErr=%.2f%%", bits, dim, rmse, relErr*100)
	}
}

func TestTurboQuantNoClip(t *testing.T) {
	dim := 8
	cfg := NewTurboQuantConfig(dim, 8)
	x := []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8}
	comp := make([]byte, cfg.CompressedSize())
	cfg.Encode(x, comp)
	out := make([]float32, dim)
	cfg.Decode(comp, out)
	t.Logf("8-bit dim=8 orig:  %v", x)
	t.Logf("8-bit dim=8 recon: %v", out)
	maxErr := float32(0)
	for i := range x {
		e := x[i] - out[i]
		if e < 0 {
			e = -e
		}
		if e > maxErr {
			maxErr = e
		}
	}
	t.Logf("max error: %f", maxErr)
}
