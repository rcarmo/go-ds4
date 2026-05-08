package ds4

import (
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// VecDotQ2KQ8K_SIMD is unused — replaced by vecDotQ2KQ8K_avx2.
func VecDotQ2KQ8K_SIMD(n int, xQ2K []byte, yQ8K []byte) float32 {
	return vecDotQ2KQ8K_avx2(n, xQ2K, yQ8K)
}

// VecDotIQ2XXSQ8K_SIMD computes IQ2_XXS · Q8_K using SIMD int8 dot.
// Unrolled grid expansion + DotI8 per 32-element group.
func VecDotIQ2XXSQ8K_SIMD(n int, xIQ2 []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sumf := float32(0)

	var gridBuf [32]int8

	for i := 0; i < nBlocks; i++ {
		xOff := i * BlockIQ2XXSSize
		yOff := i * BlockQ8KSize

		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xIQ2[xOff])))
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		scale := d * yd

		q2 := xIQ2[xOff+2:]
		q8 := yQ8K[yOff+4:]

		bsum := int32(0)

		for ib32 := 0; ib32 < QK_K/32; ib32++ {
			aux32_1 := *(*uint32)(unsafe.Pointer(&q2[ib32*8+4]))
			aux8 := (*[8]uint8)(unsafe.Pointer(&q2[ib32*8]))

			ls := int32(2*(aux32_1>>28) + 1)

			// Expand 32 grid values — unrolled
			g0 := &iq2xxsSignedGrid[int(aux8[0])*128+int(aux32_1&127)]
			g1 := &iq2xxsSignedGrid[int(aux8[1])*128+int((aux32_1>>7)&127)]
			g2 := &iq2xxsSignedGrid[int(aux8[2])*128+int((aux32_1>>14)&127)]
			g3 := &iq2xxsSignedGrid[int(aux8[3])*128+int((aux32_1>>21)&127)]

			gridBuf[0] = g0[0]; gridBuf[1] = g0[1]; gridBuf[2] = g0[2]; gridBuf[3] = g0[3]
			gridBuf[4] = g0[4]; gridBuf[5] = g0[5]; gridBuf[6] = g0[6]; gridBuf[7] = g0[7]
			gridBuf[8] = g1[0]; gridBuf[9] = g1[1]; gridBuf[10] = g1[2]; gridBuf[11] = g1[3]
			gridBuf[12] = g1[4]; gridBuf[13] = g1[5]; gridBuf[14] = g1[6]; gridBuf[15] = g1[7]
			gridBuf[16] = g2[0]; gridBuf[17] = g2[1]; gridBuf[18] = g2[2]; gridBuf[19] = g2[3]
			gridBuf[20] = g2[4]; gridBuf[21] = g2[5]; gridBuf[22] = g2[6]; gridBuf[23] = g2[7]
			gridBuf[24] = g3[0]; gridBuf[25] = g3[1]; gridBuf[26] = g3[2]; gridBuf[27] = g3[3]
			gridBuf[28] = g3[4]; gridBuf[29] = g3[5]; gridBuf[30] = g3[6]; gridBuf[31] = g3[7]

			dot := simd.DotI8(
				unsafe.Pointer(&gridBuf[0]),
				unsafe.Pointer(&q8[ib32*32]),
				32,
			)
			bsum += ls * dot
		}
		sumf += scale * float32(bsum)
	}
	return 0.125 * sumf
}
