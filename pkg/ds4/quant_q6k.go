package ds4

import "unsafe"

// VecDotQ6KQ8K computes Q6_K · Q8_K dot product.
func VecDotQ6KQ8K(n int, xQ6K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		xOff := b * 210
		yOff := b * BlockQ8KSize

		ql := xQ6K[xOff:]
		qh := xQ6K[xOff+128:]
		sc := xQ6K[xOff+192:]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ6K[xOff+208])))
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		y8 := yQ8K[yOff+4:]

		isum := int32(0)
		for j := 0; j < 32; j++ {
			q0 := (int32(ql[j]&0xF) | (int32(qh[j]&3) << 4)) - 32
			q1 := (int32(ql[j+32]&0xF) | (int32((qh[j]>>2)&3) << 4)) - 32
			q2 := (int32(ql[j]>>4) | (int32((qh[j]>>4)&3) << 4)) - 32
			q3 := (int32(ql[j+32]>>4) | (int32((qh[j]>>6)&3) << 4)) - 32

			isum += int32(int8(sc[j/16]))*q0*int32(int8(y8[j])) +
				int32(int8(sc[j/16+2]))*q1*int32(int8(y8[j+32])) +
				int32(int8(sc[j/16+4]))*q2*int32(int8(y8[j+64])) +
				int32(int8(sc[j/16+6]))*q3*int32(int8(y8[j+96]))

			q4 := (int32(ql[j+64]&0xF) | (int32(qh[j+32]&3) << 4)) - 32
			q5 := (int32(ql[j+96]&0xF) | (int32((qh[j+32]>>2)&3) << 4)) - 32
			q6 := (int32(ql[j+64]>>4) | (int32((qh[j+32]>>4)&3) << 4)) - 32
			q7 := (int32(ql[j+96]>>4) | (int32((qh[j+32]>>6)&3) << 4)) - 32

			isum += int32(int8(sc[j/16+8]))*q4*int32(int8(y8[j+128])) +
				int32(int8(sc[j/16+10]))*q5*int32(int8(y8[j+160])) +
				int32(int8(sc[j/16+12]))*q6*int32(int8(y8[j+192])) +
				int32(int8(sc[j/16+14]))*q7*int32(int8(y8[j+224]))
		}
		sum += yd * d * float32(isum)
	}
	return sum
}
