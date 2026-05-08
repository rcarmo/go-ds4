package ds4

import (
	"unsafe"
)

// Q5_K block: 176 bytes per 256 elements.
// Layout: d(f16,2B) + dmin(f16,2B) + scales(12B) + qh(32B) + qs(128B)
//
// 8 groups of 32 elements. Each element has 5 bits: 4 low bits from qs + 1 high bit from qh.
// Scales packed: 12 bytes encode 8 × (6-bit scale + 6-bit min).
//
// value = d * scale * q5 - dmin * min_val
// where q5 = (qs_nibble) | ((qh_bit) << 4)

const BlockQ5KSize = 176

// VecDotQ5KQ8K computes Q5_K · Q8_K dot product for n elements.
func VecDotQ5KQ8K(n int, xQ5K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)

	for b := 0; b < nBlocks; b++ {
		xOff := b * BlockQ5KSize
		yOff := b * BlockQ8KSize

		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ5K[xOff])))
		dmin := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ5K[xOff+2])))

		scales := xQ5K[xOff+4 : xOff+16] // 12 bytes packed scales
		qh := xQ5K[xOff+16 : xOff+48]    // 32 bytes high bits
		qs := xQ5K[xOff+48 : xOff+176]   // 128 bytes low nibbles

		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		yqs := yQ8K[yOff+4 : yOff+4+QK_K]

		// Unpack scales: 8 groups, each with 6-bit scale and 6-bit min
		// Packed into 12 bytes
		var sc [8]uint8
		var mn [8]uint8
		unpackQ5KScales(scales, sc[:], mn[:])

		sumMinS := int32(0)
		isum := int32(0)

		for j := 0; j < 8; j++ {
			scLo := int32(sc[j])
			scHi := int32(mn[j])

			// bsums: sum of Q8_K values for this 32-element group
			bs0 := *(*int16)(unsafe.Pointer(&yQ8K[yOff+4+QK_K+j*4]))
			bs1 := *(*int16)(unsafe.Pointer(&yQ8K[yOff+4+QK_K+j*4+2]))
			sumMinS += scHi * (int32(bs0) + int32(bs1))

			// Dot: 32 elements per group
			dot := int32(0)
			for k := 0; k < 32; k++ {
				elemIdx := j*32 + k

				// Low 4 bits from qs (packed as nibbles: 2 per byte)
				qsByte := qs[elemIdx/2]
				var lo uint8
				if elemIdx%2 == 0 {
					lo = qsByte & 0x0F
				} else {
					lo = qsByte >> 4
				}

				// High bit from qh (1 bit per element, packed 8 per byte)
				hi := (qh[elemIdx/8] >> (uint(elemIdx) % 8)) & 1

				// 5-bit value
				q5 := int32(lo) | (int32(hi) << 4)

				dot += q5 * int32(int8(yqs[elemIdx]))
			}
			isum += scLo * dot
		}

		sum += yd*d*float32(isum) - yd*dmin*float32(sumMinS)
	}
	return sum
}

// unpackQ5KScales extracts 8 × (6-bit scale + 6-bit min) from 12 packed bytes.
// Matches ggml's get_scale_min_k4 packing.
func unpackQ5KScales(packed []byte, scales, mins []uint8) {
	// First 4 groups: lower 6 bits from bytes 0-7
	for j := 0; j < 4; j++ {
		scales[j] = packed[j] & 63
		mins[j] = packed[j+4] & 63
	}
	// Last 4 groups: combine upper 2 bits from bytes 8-11
	for j := 0; j < 4; j++ {
		scales[j+4] = (packed[8+j] & 0xF) | ((packed[j] >> 6) << 4)
		mins[j+4] = (packed[8+j] >> 4) | ((packed[j+4] >> 6) << 4)
	}
}

func init() {
	// Register Q5_K in TensorTypeSize
	_ = BlockQ5KSize // ensure constant is used
}
