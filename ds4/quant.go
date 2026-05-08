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
	sign := (bits >> 16) & 0x8000
	exp := int32((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff

	if exp <= 0 {
		if exp < -10 {
			return uint16(sign)
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		halfMant := mant >> shift
		roundBit := (mant >> (shift - 1)) & 1
		sticky := mant & ((uint32(1) << (shift - 1)) - 1)
		if roundBit != 0 && (sticky != 0 || (halfMant&1) != 0) {
			halfMant++
		}
		return uint16(sign | halfMant)
	}

	if exp >= 31 {
		if ((bits>>23)&0xff) == 0xff && mant != 0 {
			return uint16(sign | 0x7e00)
		}
		return uint16(sign | 0x7c00)
	}

	half := sign | (uint32(exp) << 10) | (mant >> 13)
	round := mant & 0x1fff
	if round > 0x1000 || (round == 0x1000 && (half&1) != 0) {
		half++
	}
	return uint16(half)
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

// vecDotQ2KQ8K_scalar matches ds4.c ds4_vec_dot_q2_K_q8_K exactly.
func vecDotQ2KQ8K_scalar(n int, xQ2K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)

	for b := 0; b < nBlocks; b++ {
		xOff := b * BlockQ2KSize
		yOff := b * BlockQ8KSize

		sc := xQ2K[xOff : xOff+16]
		q2 := xQ2K[xOff+16 : xOff+16+64]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ2K[xOff+80])))
		dmin := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ2K[xOff+82])))

		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		q8 := yQ8K[yOff+4 : yOff+4+QK_K]
		bsums := yQ8K[yOff+4+QK_K : yOff+4+QK_K+32]

		summs := int32(0)
		for j := 0; j < 16; j++ {
			bs := *(*int16)(unsafe.Pointer(&bsums[j*2]))
			summs += int32(bs) * int32(sc[j]>>4)
		}

		isum := int32(0)
		is := 0
		q2off := 0
		q8off := 0
		for k := 0; k < QK_K/128; k++ {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dscale := int32(sc[is] & 0x0f)
				is++
				isum += dscale * dotQ2K16(q2[q2off:q2off+16], q8[q8off:q8off+16], shift)

				dscale = int32(sc[is] & 0x0f)
				is++
				isum += dscale * dotQ2K16(q2[q2off+16:q2off+32], q8[q8off+16:q8off+32], shift)

				shift += 2
				q8off += 32
			}
			q2off += 32
		}

		sum += yd*d*float32(isum) - yd*dmin*float32(summs)
	}
	return sum
}

func dotQ2K16(q2 []byte, q8 []byte, shift uint) int32 {
	var sum int32
	for i := 0; i < 16; i++ {
		q := int32((q2[i] >> shift) & 0x03)
		sum += q * int32(int8(q8[i]))
	}
	return sum
}

// dequantEmbedding extracts one row of quantized embedding data to float32.
func DequantEmbedding(out []float32, data []byte, typ uint32, n int) {
	switch typ {
	case TensorQ2_K:
		// Q2_K: extract using scales + 2-bit values
		nBlocks := n / QK_K
		for b := 0; b < nBlocks; b++ {
			off := b * BlockQ2KSize
			scales := data[off : off+16]
			qs := data[off+16 : off+80]
			d := F16ToF32(*(*uint16)(unsafe.Pointer(&data[off+80])))
			dmin := F16ToF32(*(*uint16)(unsafe.Pointer(&data[off+82])))
			for j := 0; j < 16; j++ {
				sc := scales[j]
				scLo := float32(sc & 0x0f)
				scHi := float32(sc >> 4)
				for k := 0; k < 4; k++ {
					qb := qs[j*4+k]
					for m := 0; m < 4; m++ {
						q := float32((qb >> (uint(m) * 2)) & 3)
						idx := b*QK_K + j*16 + k*4 + m
						if idx < n {
							out[idx] = d*scLo*q - dmin*scHi
						}
					}
				}
			}
		}
	case TensorF32:
		f32 := tensorF32Unsafe(data)
		copy(out, f32[:n])
	case TensorF16:
		u16 := tensorU16Unsafe(data)
		for i := 0; i < n; i++ {
			out[i] = F16ToF32(u16[i])
		}
	default:
		for i := range out[:n] {
			out[i] = 0
		}
	}
}
