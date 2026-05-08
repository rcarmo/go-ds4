package ds4

import (
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// VecDotQ2KQ8K_SIMD computes Q2_K · Q8_K using SIMD where possible.
// The per-group structure (16 groups × 16 elements with different scales)
// prevents a single bulk DotI8, but we use SIMD for each 16-element sub-dot.
func VecDotQ2KQ8K_SIMD(n int, xQ2K []byte, yQ8K []byte) float32 {
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

			// Expand 16 Q2 values from packed bytes
			var q2vals [16]int8
			for i := 0; i < 16; i++ {
				idx := j*16 + i
				byteIdx := idx / 4
				bitShift := uint(idx%4) * 2
				q2vals[i] = int8((qs[byteIdx] >> bitShift) & 3)
			}

			// Scalar dot for 16 elements (DotI8 needs 32+ for SIMD benefit)
			dot := int32(0)
			for i := 0; i < 16; i++ {
				dot += int32(q2vals[i]) * int32(int8(yqs[j*16+i]))
			}
			isum += scLo * dot
		}

		sum += yd*d*float32(isum) - yd*dmin*float32(sumMinS)
	}
	return sum
}

// VecDotIQ2XXSQ8K_SIMD computes IQ2_XXS · Q8_K using SIMD int8 dot.
// Each 32-element group has one scale (ls), so we expand all 32 grid values
// and use DotI8 on the full 32-element chunk.
func VecDotIQ2XXSQ8K_SIMD(n int, xIQ2 []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sumf := float32(0)

	for i := 0; i < nBlocks; i++ {
		xOff := i * BlockIQ2XXSSize
		yOff := i * BlockQ8KSize

		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xIQ2[xOff])))
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		scale := d * yd

		q2 := xIQ2[xOff+2:]
		q8 := yQ8K[yOff+4:]

		bsum := int32(0)

		// 8 groups of 32 elements
		for ib32 := 0; ib32 < QK_K/32; ib32++ {
			aux32_1 := *(*uint32)(unsafe.Pointer(&q2[ib32*8+4]))
			aux8 := (*[8]uint8)(unsafe.Pointer(&q2[ib32*8]))

			ls := int32(2*(aux32_1>>28) + 1)

			// Expand all 32 grid values into contiguous int8 array
			var gridBuf [32]int8
			for l := 0; l < 4; l += 2 {
				gridIdx0 := aux8[l]
				gridIdx1 := aux8[l+1]
				signIdx0 := (aux32_1 >> (7 * uint(l))) & 127
				signIdx1 := (aux32_1 >> (7 * uint(l+1))) & 127

				grid0 := &iq2xxsSignedGrid[int(gridIdx0)*128+int(signIdx0)]
				grid1 := &iq2xxsSignedGrid[int(gridIdx1)*128+int(signIdx1)]

				for k := 0; k < 8; k++ {
					gridBuf[l*8+k] = grid0[k]
					gridBuf[l*8+8+k] = grid1[k]
				}			}

			// SIMD DotI8 on 32 elements — fits exactly one AVX2 register
			q8Off := ib32 * 32
			dot := simd.DotI8(
				unsafe.Pointer(&gridBuf[0]),
				unsafe.Pointer(&q8[q8Off]),
				32,
			)
			bsum += ls * dot
		}
		sumf += scale * float32(bsum)
	}
	return 0.125 * sumf
}
