package ds4

import "unsafe"

const BlockQ3KSize = 110

func VecDotQ3KQ8K(n int, xQ3K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		xOff := b * BlockQ3KSize
		yOff := b * BlockQ8KSize
		hmask := xQ3K[xOff : xOff+32]
		qs := xQ3K[xOff+32 : xOff+96]
		scales := xQ3K[xOff+96 : xOff+108]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ3K[xOff+108])))
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		yqs := yQ8K[yOff+4 : yOff+4+QK_K]

		// Unpack scales (12 bytes → 16 groups, each 6-bit signed)
		var sc [16]int8
		for j := 0; j < 16; j++ {
			var raw int
			if j < 8 {
				raw = int(scales[j])
			} else {
				// Upper 8 scales packed in remaining bytes
				idx := j - 8
				raw = int(scales[idx]) >> 4
				if idx < 4 {
					raw = int(scales[8+idx/2])
					if idx%2 == 0 {
						raw &= 0xF
					} else {
						raw >>= 4
					}
				}
			}
			// Sign extend 6-bit to int8
			if raw >= 32 {
				raw -= 64
			}
			sc[j] = int8(raw)
		}

		isum := int32(0)
		for j := 0; j < QK_K; j++ {
			group := j / 16
			// Low 2 bits from qs (packed 4 per byte)
			qsByte := qs[j/4]
			lo := (qsByte >> (uint(j%4) * 2)) & 3
			// High bit from hmask
			hi := (hmask[j/8] >> (uint(j) % 8)) & 1
			q3 := int32(lo) | (int32(hi) << 2)
			q3 -= 4 // center
			isum += int32(sc[group]) * q3 * int32(int8(yqs[j]))
		}
		sum += yd * d * float32(isum)
	}
	return sum
}
