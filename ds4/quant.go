package ds4

import (
	"math"
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// F16 conversion — matches ds4.c f16_to_f32 / f32_to_f16.

func F16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
	}
	if exp == 31 {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000) // inf
		}
		return math.Float32frombits(sign | 0x7fc00000) // nan
	}
	return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

func F32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp)<<10 | uint16(mant>>13)
}

// Q8_0: 32-element blocks. Used for attention projections, shared experts, output.
// DS4 layout: f16 scale (2 bytes) + int8[32] quantized values = 34 bytes.

// DotQ8_0F32 computes dot(Q8_0 weight row, float32 activation) for n elements.
// Uses SIMD on amd64/arm64 via simd.DotQ8_0F32 (DS4 34-byte layout).
func DotQ8_0F32(wq8 []byte, x []float32, n int) float32 {
	nBlocks := n / 32
	return simd.DotQ8_0F32(unsafe.Pointer(&wq8[0]), unsafe.Pointer(&x[0]), nBlocks)
}

// QuantizeRowQ8K quantizes a float32 row to Q8_K blocks.
// Uses fast rounding without math.Round.
func QuantizeRowQ8K(x []float32, out []byte) {
	simd.QuantizeQ8K_fast(x, out)
}

// VecDotQ2KQ8K computes Q2_K · Q8_K dot product.
// Uses AVX2 DotI8 on pre-scaled expanded Q2 values.
func VecDotQ2KQ8K(n int, xQ2K []byte, yQ8K []byte) float32 {
	return vecDotQ2KQ8K_scalar(n, xQ2K, yQ8K)
}

// vecDotQ2KQ8K_scalar is the optimized implementation using AVX2 DotQ2x32
// for the inner 128-element dot product chunks.
func vecDotQ2KQ8K_scalar(n int, xQ2K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)

	for b := 0; b < nBlocks; b++ {
		xOff := b * BlockQ2KSize
		yOff := b * BlockQ8KSize

		scales := xQ2K[xOff : xOff+16]
		qs := xQ2K[xOff+16 : xOff+16+64]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ2K[xOff+80])))
		dmin := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ2K[xOff+82])))

		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		yqs := yQ8K[yOff+4 : yOff+4+QK_K]
		ybsums := yQ8K[yOff+4+QK_K : yOff+4+QK_K+32]

		sumMinS := int32(0)
		isum := int32(0)

		for j := 0; j < 16; j++ {
			sc := scales[j]
			scHi := int32(sc >> 4)
			scLo := int32(sc & 0x0f)

			bs := *(*int16)(unsafe.Pointer(&ybsums[j*2]))
			sumMinS += scHi * int32(bs)

			// AVX2: unpack 16 Q2 values + dot with 16 Q8 values
			dot := simd.DotQ2Group16(
				unsafe.Pointer(&qs[j*4]),
				unsafe.Pointer(&yqs[j*16]),
			)
			isum += scLo * dot
		}

		sum += yd*d*float32(isum) - yd*dmin*float32(sumMinS)
	}
	return sum
}
