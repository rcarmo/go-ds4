package ds4

import "unsafe"

// Q6_K: 6-bit quantization, 256 elements per block, 210 bytes.
// Layout: ql[128] + qh[64] + scales[16] + d(f16,2)
func VecDotQ6KQ8K(n int, xQ6K []byte, yQ8K []byte) float32 {
	nBlocks := n / QK_K
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		xOff := b * 210
		yOff := b * BlockQ8KSize
		ql := xQ6K[xOff : xOff+128]
		qh := xQ6K[xOff+128 : xOff+192]
		sc := xQ6K[xOff+192 : xOff+208]
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&xQ6K[xOff+208])))
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		yqs := yQ8K[yOff+4 : yOff+4+QK_K]
		isum := int32(0)
		for j := 0; j < 16; j++ {
			scale := int32(int8(sc[j]))
			dot := int32(0)
			for k := 0; k < 16; k++ {
				idx := j*16 + k
				lo := ql[idx/2]
				var q4 uint8
				if idx%2 == 0 {
					q4 = lo & 0xF
				} else {
					q4 = lo >> 4
				}
				hi := (qh[idx/4] >> (uint(idx%4) * 2)) & 3
				q6 := int32(q4) | (int32(hi) << 4)
				q6 -= 32
				dot += q6 * int32(int8(yqs[idx]))
			}
			isum += scale * dot
		}
		sum += yd * d * float32(isum)
	}
	return sum
}
