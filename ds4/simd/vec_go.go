package simd

import "math"

// vecSiLUMulGo is the Go scalar implementation called by the asm stubs.
func vecSiLUMulGo(dst, a, b []float32) {
	for i := range dst {
		x := a[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		dst[i] = s * b[i]
	}
}
