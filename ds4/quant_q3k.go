package ds4

import "unsafe"

// Q3_K: 3-bit quantization, 256 elements per block, 110 bytes.
// Layout: hmask[32] + qs[64] + scales[12] + d(f16,2)
//
// 256 elements in 16 groups of 16.
// Each element: q3 = (qs 2-bit) | (hmask 1-bit << 2), range [0,7]
// Effective value: d * (q3 - 4) * scale
//
// Scale packing (12 bytes → 16 signed 6-bit scales):
//   Bytes 0-3: scales[0..7] low 4 bits (packed 2 per byte, low/high nibble)
//   Bytes 4-7: scales[8..15] low 4 bits
//   Bytes 8-11: high 2 bits for all 16 scales (packed 4 per byte)

const BlockQ3KSize = 110

func VecDotQ3KQ8K(n int, xQ3K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)

	for b := 0; b < nBlocks; b++ {
		xOff := b * BlockQ3KSize
		yOff := b * BlockQ8KSize

		hmask := xQ3K[xOff : xOff+32]
		qs := xQ3K[xOff+32 : xOff+96]
		scBytes := xQ3K[xOff+96 : xOff+108]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ3K[xOff+108])))
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		yqs := yQ8K[yOff+4 : yOff+4+QK_K]
		ybsums := yQ8K[yOff+4+QK_K : yOff+4+QK_K+32]

		// Unpack 16 scales from 12 bytes (6-bit signed each)
		var sc [16]int32
		// Low 4 bits: bytes 0-3 hold scales 0-7, bytes 4-7 hold scales 8-15
		for j := 0; j < 4; j++ {
			sc[2*j] = int32(scBytes[j] & 0xF)
			sc[2*j+1] = int32(scBytes[j] >> 4)
			sc[8+2*j] = int32(scBytes[4+j] & 0xF)
			sc[8+2*j+1] = int32(scBytes[4+j] >> 4)
		}
		// High 2 bits: bytes 8-11, each byte has 4 × 2-bit values
		for j := 0; j < 4; j++ {
			hb := scBytes[8+j]
			sc[4*j] |= int32(hb&3) << 4
			sc[4*j+1] |= int32((hb>>2)&3) << 4
			sc[4*j+2] |= int32((hb>>4)&3) << 4
			sc[4*j+3] |= int32(hb>>6) << 4
		}
		// Sign extend 6-bit to signed
		for j := range sc {
			if sc[j] >= 32 {
				sc[j] -= 64
			}
		}

		// Compute: sum += d * yd * Σ_j sc[j] * (dot(q3[j*16:(j+1)*16], yqs[j*16:(j+1)*16]) - 4 * bsum[j])
		isum := int32(0)
		for j := 0; j < 16; j++ {
			// Dot product of 16 Q3 values × Q8 values
			dot := int32(0)
			for k := 0; k < 16; k++ {
				elemIdx := j*16 + k
				// Low 2 bits from qs (packed 4 per byte)
				qsByte := qs[elemIdx/4]
				lo := (qsByte >> (uint(elemIdx%4) * 2)) & 3
				// High bit from hmask
				hi := (hmask[elemIdx/8] >> (uint(elemIdx) % 8)) & 1
				q3 := int32(lo) | (int32(hi) << 2) // [0,7]
				dot += q3 * int32(int8(yqs[elemIdx]))
			}
			// bsums: sum of Q8_K values for this 16-element group
			bs := *(*int16)(unsafe.Pointer(&ybsums[j*2]))
			// value = d * sc[j] * (q3 - 4) * y = d * sc[j] * (q3*y - 4*sum(y))
			isum += sc[j] * (dot - 4*int32(bs))
		}

		sum += yd * d * float32(isum)
	}
	return sum
}
