package ds4

import (
	"unsafe"

	"github.com/rcarmo/go-ds4/pkg/ds4/internal/simd"
)

// VecDotIQ2XXSQ8K computes the dot product of IQ2_XXS weights with Q8_K activations.
// This is the hot path for routed expert gate/up projections.
//
// IQ2_XXS block (66 bytes per 256 elements):
//
//	d: uint16 (F16 scale)
//	qs: uint16[32] — packed: each group of 4 uint16 encodes 32 values:
//	  - qs[0..3] as 2 uint32 (aux32):
//	    - aux8[0], aux8[1]: grid indices (0-255) → 8 values each
//	    - aux32[1] bits 0-6, 7-13: sign indices (0-127) for the two grids
//	    - aux32[1] bits 14-20, 21-27: sign indices for next two grids
//	    - aux32[1] bits 28-31: 4-bit scale (ls = 2*nibble + 1)
//
// Q8_K block (292 bytes per 256 elements):
//
//	d: float32 scale
//	qs: int8[256] quantized values
//	bsums: int16[16] partial sums
//
// VecDotIQ2XXSQ8K computes IQ2_XXS · Q8_K using SIMD integer dot products.
func VecDotIQ2XXSQ8K(n int, xIQ2 []byte, yQ8K []byte) float32 {
	return VecDotIQ2XXSQ8K_SIMD(n, xIQ2, yQ8K)
}

// vecDotIQ2XXSQ8K_scalar is the reference scalar implementation.
func vecDotIQ2XXSQ8K_scalar(n int, xIQ2 []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sumf := float32(0)

	for i := 0; i < nBlocks; i++ {
		xOff := i * BlockIQ2XXSSize
		yOff := i * BlockQ8KSize

		// IQ2_XXS: d(f16) + qs[32]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xIQ2[xOff])))
		// Q8_K: d(f32) + qs[256] + bsums[16]
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))

		scale := d * yd
		q2 := xIQ2[xOff+2:] // uint16 array as bytes
		q8 := yQ8K[yOff+4:] // int8 array

		bsum := int32(0)

		// 8 groups of 32 values = 256 total
		for ib32 := 0; ib32 < QK_K/32; ib32++ {
			// Read 2 uint32 from q2 (which is uint16[] packed as bytes)
			aux32_1 := *(*uint32)(unsafe.Pointer(&q2[ib32*8+4]))
			aux8 := (*[8]uint8)(unsafe.Pointer(&q2[ib32*8]))

			// Scale for this group of 32
			ls := int32(2*(aux32_1>>28) + 1)

			sumi := int32(0)
			// 4 sub-groups of 8 values, processed in pairs of 2
			for l := 0; l < 4; l += 2 {
				gridIdx0 := aux8[l]
				gridIdx1 := aux8[l+1]
				signIdx0 := (aux32_1 >> (7 * uint(l))) & 127
				signIdx1 := (aux32_1 >> (7 * uint(l+1))) & 127

				// Lookup signed grid values
				grid0 := &iq2xxsSignedGrid[int(gridIdx0)*128+int(signIdx0)]
				grid1 := &iq2xxsSignedGrid[int(gridIdx1)*128+int(signIdx1)]

				// Dot product: 8 grid values × 8 Q8 values, twice
				q8Off := (ib32*32 + l*8) // offset into Q8 values
				sum := int32(0)
				for k := 0; k < 8; k++ {
					sum += int32(grid0[k]) * int32(int8(q8[q8Off+k]))
				}
				for k := 0; k < 8; k++ {
					sum += int32(grid1[k]) * int32(int8(q8[q8Off+8+k]))
				}
				sumi += sum
			}
			bsum += sumi * ls
		}
		sumf += scale * float32(bsum)
	}
	return 0.125 * sumf
}

// VecDotIQ2XXSPairQ8K computes two IQ2_XXS dot products simultaneously
// against the same Q8_K activation. Used for fused gate+up expert projections.
func VecDotIQ2XXSPairQ8K(n int, x0IQ2, x1IQ2 []byte, yQ8K []byte) (float32, float32) {
	// For the scalar path, just call the single dot twice.
	// The SIMD path will fuse these for better throughput.
	s0 := VecDotIQ2XXSQ8K(n, x0IQ2, yQ8K)
	s1 := VecDotIQ2XXSQ8K(n, x1IQ2, yQ8K)
	return s0, s1
}

// DotF16 computes dot product of F16 weight row with float32 activation.
// Uses SIMD (VCVTPH2PS on amd64, FCVTL on arm64) when available.
func DotF16(wF16 []uint16, x []float32) float32 {
	return simd.DotF16F32(unsafe.Pointer(&wF16[0]), unsafe.Pointer(&x[0]), len(wF16))
}
