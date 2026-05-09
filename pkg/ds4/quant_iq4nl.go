package ds4

import "unsafe"

var iq4nlTable = [16]int8{
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
}

func VecDotIQ4NLF32(wIQ4 []byte, x []float32, n int) float32 {
	nBlocks := (n + QK4_NL - 1) / QK4_NL
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		off := b * BlockIQ4NLSize
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&wIQ4[off])))
		dot := float32(0)
		xOff := b * QK4_NL
		remaining := n - xOff
		if remaining > QK4_NL {
			remaining = QK4_NL
		}
		for i := 0; i < remaining; i += 2 {
			qsByte := wIQ4[off+2+i/2]
			dot += float32(iq4nlTable[qsByte&0xF]) * x[xOff+i]
			if i+1 < remaining {
				dot += float32(iq4nlTable[qsByte>>4]) * x[xOff+i+1]
			}
		}
		sum += d * dot
	}
	return sum
}

// VecDotIQ4NLQ8K dots IQ4_NL weight directly against Q8_K activation (no F32 intermediate).
func VecDotIQ4NLQ8K(n int, wIQ4 []byte, yQ8K []byte) float32 {
	nBlocksIQ4 := (n + QK4_NL - 1) / QK4_NL
	sum := float32(0)
	for b := 0; b < nBlocksIQ4; b++ {
		off := b * BlockIQ4NLSize
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&wIQ4[off])))
		xOff := b * QK4_NL
		// Find which Q8K block this falls in
		q8Block := xOff / QK_K
		q8Off := q8Block * BlockQ8KSize
		q8d := *(*float32)(unsafe.Pointer(&yQ8K[q8Off]))
		q8qs := yQ8K[q8Off+4:]
		q8Elem := xOff - q8Block*QK_K

		dot := int32(0)
		remaining := n - xOff
		if remaining > QK4_NL {
			remaining = QK4_NL
		}
		for i := 0; i < remaining; i += 2 {
			qsByte := wIQ4[off+2+i/2]
			dot += int32(iq4nlTable[qsByte&0xF]) * int32(int8(q8qs[q8Elem+i]))
			if i+1 < remaining {
				dot += int32(iq4nlTable[qsByte>>4]) * int32(int8(q8qs[q8Elem+i+1]))
			}
		}
		sum += d * q8d * float32(dot)
	}
	return sum
}
