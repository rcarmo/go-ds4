package simd

// vecSiLUMulGo is the Go implementation called by the asm stubs.
// Uses fast exp approximation instead of math.Exp.
func vecSiLUMulGo(dst, a, b []float32) {
	for i := range dst {
		x := a[i]
		sig := fastSigmoid(x)
		dst[i] = x * sig * b[i]
	}
}
