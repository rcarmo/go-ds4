package ds4

import (
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// vecDotQ2KQ8K_avx2 pre-scales Q2 values by their group scale,
// then uses SIMD DotI8 for the full 256-element block.
func vecDotQ2KQ8K_avx2(n int, xQ2K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)

	var q2Scaled [QK_K]int8 // pre-scaled Q2 values

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

		// Expand + pre-scale all 256 Q2 values
		for j := 0; j < 16; j++ {
			sc := scales[j]
			scHi := int32(sc >> 4)
			scLo := int8(sc & 0x0f)

			bs := *(*int16)(unsafe.Pointer(&ybsums[j*2]))
			sumMinS += scHi * int32(bs)

			// Expand 16 Q2 values and multiply by scLo
			for k := 0; k < 4; k++ {
				qb := qs[j*4+k]
				base := j*16 + k*4
				q2Scaled[base] = scLo * int8(qb&3)
				q2Scaled[base+1] = scLo * int8((qb>>2)&3)
				q2Scaled[base+2] = scLo * int8((qb>>4)&3)
				q2Scaled[base+3] = scLo * int8(qb>>6)
			}
		}

		// One SIMD dot for the full 256-element block
		isum := simd.DotI8(
			unsafe.Pointer(&q2Scaled[0]),
			unsafe.Pointer(&yqs[0]),
			QK_K,
		)

		sum += yd*d*float32(isum) - yd*dmin*float32(sumMinS)
	}
	return sum
}
