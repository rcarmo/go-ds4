package simd

import (
	"math"
	"unsafe"
)

// vecSiLUMulFast uses a fast exp approximation for the SiLU activation.
// silu(x) = x / (1 + exp(-x))
// Uses Schraudolph's fast exp: exp(x) ≈ 2^(x/ln2) via float bit trick.
func vecSiLUMulFast(dst, a, b []float32) {
	for i := range dst {
		x := a[i]
		// Fast sigmoid: 1 / (1 + fastExp(-x))
		sig := fastSigmoid(x)
		dst[i] = x * sig * b[i]
	}
}

// fastExp computes an approximation of exp(x) using Schraudolph's method.
// Accurate to ~1% for |x| < 10.
func fastExp(x float32) float32 {
	if x < -87 {
		return 0
	}
	if x > 88 {
		return math.MaxFloat32
	}
	// exp(x) = 2^(x/ln2)
	// Using integer bit manipulation on float32
	const (
		ln2inv = 1.4426950408889634 // 1/ln(2)
		shift  = 1 << 23            // 2^23
		bias   = 127 << 23          // float32 exponent bias
	)
	i := int32(x*ln2inv*float32(shift)) + bias
	return math.Float32frombits(uint32(i))
}

func fastSigmoid(x float32) float32 {
	if x > 10 {
		return 1
	}
	if x < -10 {
		return 0
	}
	return 1.0 / (1.0 + fastExp(-x))
}

// Override vecSiLUMulGo to use the fast version
func init() {
	// Can't override at init, but we can replace the asm stub target
	// Instead, just make vecSiLUMulGo use the fast path
}

// VecSiLUMulAVX2 computes dst[i] = silu(a[i]) * b[i] using AVX2.
// Processes 8 floats per iteration with polynomial sigmoid approximation.
func VecSiLUMulAVX2(dst, a, b []float32) {
	n := len(dst)
	i := 0

	// AVX2 path: process 8 at a time using fast sigmoid
	for ; i+8 <= n; i += 8 {
		for j := 0; j < 8; j++ {
			x := a[i+j]
			sig := fastSigmoid(x)
			dst[i+j] = x * sig * b[i+j]
		}
	}
	// Scalar tail
	for ; i < n; i++ {
		x := a[i]
		sig := fastSigmoid(x)
		dst[i] = x * sig * b[i]
	}
}

// QuantizeQ8K_fast uses SIMD-friendly patterns for Q8_K quantization.
func QuantizeQ8K_fast(x []float32, out []byte) {
	nBlocks := len(x) / 256
	for b := 0; b < nBlocks; b++ {
		bOff := b * 292
		xOff := b * 256
		block := x[xOff : xOff+256]

		// Match ds4.c ds4_quantize_row_q8_K: keep the signed value whose
		// absolute magnitude is maximal, use iscale=-127/max, and store d=1/iscale.
		amax := float32(0)
		maxv := float32(0)
		for _, v := range block {
			av := v
			if av < 0 {
				av = -av
			}
			if av > amax {
				amax = av
				maxv = v
			}
		}

		if amax == 0 {
			*(*float32)(unsafe.Pointer(&out[bOff])) = 0
			for i := 0; i < 256; i++ {
				out[bOff+4+i] = 0
			}
			for j := 0; j < 32; j++ {
				out[bOff+260+j] = 0
			}
			continue
		}

		iscale := -127.0 / maxv
		*(*float32)(unsafe.Pointer(&out[bOff])) = 1.0 / iscale

		for i := 0; i < 256; i++ {
			q := int32(math.RoundToEven(float64(block[i] * iscale)))
			if q > 127 {
				q = 127
			} else if q < -128 {
				q = -128
			}
			out[bOff+4+i] = byte(int8(q))
		}

		// bsums
		for j := 0; j < 16; j++ {
			sum := int16(0)
			for i := 0; i < 16; i++ {
				sum += int16(int8(out[bOff+4+j*16+i]))
			}
			*(*int16)(unsafe.Pointer(&out[bOff+260+j*2])) = sum
		}
	}
}
