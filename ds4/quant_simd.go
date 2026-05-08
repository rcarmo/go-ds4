package ds4

import (
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// VecDotQ2KQ8K_SIMD is unused — replaced by vecDotQ2KQ8K_scalar (which uses AVX2 DotQ2Group16).
func VecDotQ2KQ8K_SIMD(n int, xQ2K []byte, yQ8K []byte) float32 {
	return vecDotQ2KQ8K_avx2(n, xQ2K, yQ8K)
}

// VecDotIQ2XXSQ8K_SIMD computes IQ2_XXS · Q8_K using AVX2 DotIQ2Group32.
// Zero-copy: passes grid pointers directly to the AVX2 kernel.
// No intermediate gridBuf allocation or copy.
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

		for ib32 := 0; ib32 < QK_K/32; ib32++ {
			aux32_1 := *(*uint32)(unsafe.Pointer(&q2[ib32*8+4]))
			aux8 := (*[8]uint8)(unsafe.Pointer(&q2[ib32*8]))

			ls := int32(2*(aux32_1>>28) + 1)

			// Direct grid pointers — no copy
			g0 := unsafe.Pointer(&iq2xxsSignedGrid[int(aux8[0])*128+int(aux32_1&127)])
			g1 := unsafe.Pointer(&iq2xxsSignedGrid[int(aux8[1])*128+int((aux32_1>>7)&127)])
			g2 := unsafe.Pointer(&iq2xxsSignedGrid[int(aux8[2])*128+int((aux32_1>>14)&127)])
			g3 := unsafe.Pointer(&iq2xxsSignedGrid[int(aux8[3])*128+int((aux32_1>>21)&127)])

			// AVX2: load 4×8 grid bytes + 32 Q8 bytes, dot in one call
			dot := simd.DotIQ2Group32(g0, g1, g2, g3,
				unsafe.Pointer(&q8[ib32*32]))

			bsum += ls * dot
		}
		sumf += scale * float32(bsum)
	}
	return 0.125 * sumf
}
