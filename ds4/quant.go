package ds4

import (
	"math"
	"unsafe"
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
		return math.Float32frombits(sign | ((exp+127-15)<<23) | (mant << 13))
	}
	if exp == 31 {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000) // inf
		}
		return math.Float32frombits(sign | 0x7fc00000) // nan
	}
	return math.Float32frombits(sign | ((exp+127-15)<<23) | (mant << 13))
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
// Layout: float32 scale (4 bytes) + int8[32] quantized values = 36 bytes.

// DotQ8_0F32 computes dot(Q8_0 weight row, float32 activation) for n elements.
// wq8 is packed Q8_0 blocks, x is float32.
func DotQ8_0F32(wq8 []byte, x []float32, n int) float32 {
	const blockSize = 32
	nBlocks := n / blockSize
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		off := b * BlockQ8_0Size
		scale := *(*float32)(unsafe.Pointer(&wq8[off]))
		qs := wq8[off+4 : off+4+blockSize]
		xOff := b * blockSize
		bsum := int32(0)
		for i := 0; i < blockSize; i++ {
			bsum += int32(int8(qs[i])) * int32(math.Float32bits(x[xOff+i])>>0) // wrong — need proper approach
		}
		// Actually: Q8_0 dot is: sum += scale * Σ(qs[i] * x[i])
		// But x is float32, not int8. So we need: scale * Σ(int8(qs[i]) * x[i])
		bsum = 0
		var fsum float32
		for i := 0; i < blockSize; i++ {
			fsum += float32(int8(qs[i])) * x[xOff+i]
		}
		sum += scale * fsum
	}
	return sum
}

// QuantizeRowQ8K quantizes a float32 row to Q8_K blocks.
// Each Q8_K block: float32 d + int8[256] qs + int16[16] bsums = 292 bytes.
func QuantizeRowQ8K(x []float32, out []byte) {
	nBlocks := len(x) / QK_K
	for b := 0; b < nBlocks; b++ {
		bOff := b * BlockQ8KSize
		xOff := b * QK_K

		// Find absmax for this block
		amax := float32(0)
		for i := 0; i < QK_K; i++ {
			v := x[xOff+i]
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}

		// Scale
		d := amax / 127.0
		*(*float32)(unsafe.Pointer(&out[bOff])) = d

		var id float32
		if d != 0 {
			id = 1.0 / d
		}

		// Quantize and compute bsums
		qs := out[bOff+4 : bOff+4+QK_K]
		bsums := out[bOff+4+QK_K : bOff+4+QK_K+32]
		for j := 0; j < 16; j++ {
			sum := int16(0)
			for i := 0; i < 16; i++ {
				idx := j*16 + i
				q := int8(math.Round(float64(x[xOff+idx] * id)))
				qs[idx] = byte(q)
				sum += int16(q)
			}
			*(*int16)(unsafe.Pointer(&bsums[j*2])) = sum
		}
	}
}

// VecDotQ2KQ8K computes the dot product of Q2_K weights with Q8_K quantized activations.
// This is the hot path for routed expert down-projections.
func VecDotQ2KQ8K(n int, xQ2K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)

	for b := 0; b < nBlocks; b++ {
		xOff := b * BlockQ2KSize
		yOff := b * BlockQ8KSize

		// Q2_K block: scales[16] + qs[64] + d(f16) + dmin(f16)
		scales := xQ2K[xOff : xOff+16]
		qs := xQ2K[xOff+16 : xOff+16+64]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ2K[xOff+80])))
		dmin := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ2K[xOff+82])))

		// Q8_K block: d(f32) + qs[256] + bsums[16]
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		yqs := yQ8K[yOff+4 : yOff+4+QK_K]
		ybsums := yQ8K[yOff+4+QK_K : yOff+4+QK_K+32]

		// Accumulate using the Q2_K structure:
		// 256 elements in 16 groups of 16, each with a 4-bit scale
		sumMinS := int32(0)
		isum := int32(0)

		for j := 0; j < 16; j++ {
			sc := scales[j]
			scHi := int32(sc >> 4)    // upper nibble = min scale
			scLo := int32(sc & 0x0f)  // lower nibble = weight scale

			// bsums for min correction
			bs := int16(*(*int16)(unsafe.Pointer(&ybsums[j*2])))
			sumMinS += scHi * int32(bs)

			// Dot product of 16 Q2 values with Q8 values
			var dsum int32
			for i := 0; i < 16; i++ {
				idx := j*16 + i
				// Extract 2-bit value from packed qs
				byteIdx := idx / 4
				bitShift := uint(idx%4) * 2
				q2val := int32((qs[byteIdx] >> bitShift) & 3)
				q8val := int32(int8(yqs[idx]))
				dsum += q2val * q8val
			}
			isum += scLo * dsum
		}

		sum += yd*d*float32(isum) - yd*dmin*float32(sumMinS)
	}
	return sum
}
