package ds4

import (
	"unsafe"
)

// IQ4_NL: 4-bit quantization with non-linear levels.
// Block: 32 elements, 18 bytes = f16 scale (2B) + packed 4-bit values (16B).
// Each byte holds 2 values: low nibble + high nibble.
// Values are mapped through a 16-entry lookup table (non-linear levels).

// IQ4NL lookup table (from ggml)
var iq4nlTable = [16]int8{
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
}

// VecDotIQ4NLF32 computes dot(IQ4_NL weight, float32 activation) for n elements.
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

		for i := 0; i < remaining; i++ {
			qsByte := wIQ4[off+2+i/2]
			var nibble uint8
			if i%2 == 0 {
				nibble = qsByte & 0x0F
			} else {
				nibble = qsByte >> 4
			}
			val := iq4nlTable[nibble]
			dot += float32(val) * x[xOff+i]
		}
		sum += d * dot
	}
	return sum
}

// VecDotIQ4NLQ8K computes dot(IQ4_NL weight, Q8_K activation) for n elements.
// Used for expert down projections in V2 Lite.
func VecDotIQ4NLQ8K(n int, wIQ4 []byte, yQ8K []byte) float32 {
	// For simplicity: dequantize Q8_K to float32, then use float dot
	nBlocksQ8 := n / QK_K
	xf32 := make([]float32, n)
	for b := 0; b < nBlocksQ8; b++ {
		yOff := b * BlockQ8KSize
		yd := *(*float32)(unsafe.Pointer(&yQ8K[yOff]))
		for i := 0; i < QK_K; i++ {
			xf32[b*QK_K+i] = yd * float32(int8(yQ8K[yOff+4+i]))
		}
	}
	return VecDotIQ4NLF32(wIQ4, xf32, n)
}
