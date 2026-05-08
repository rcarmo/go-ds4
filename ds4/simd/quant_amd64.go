//go:build amd64

package simd

import (
	"math"
	"unsafe"
)

// quantizeQ8KGo is the Go scalar implementation called by the amd64 asm stub.
func quantizeQ8KGo(x unsafe.Pointer, out unsafe.Pointer, n int) {
	xp := (*[1 << 30]float32)(x)
	op := (*[1 << 30]byte)(out)
	nBlocks := n / 256
	for b := 0; b < nBlocks; b++ {
		bOff := b * 292
		xOff := b * 256
		amax := float32(0)
		for i := 0; i < 256; i++ {
			v := xp[xOff+i]
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}
		d := amax / 127.0
		*(*float32)(unsafe.Pointer(&op[bOff])) = d
		var id float32
		if d != 0 {
			id = 1.0 / d
		}
		for i := 0; i < 256; i++ {
			q := int8(math.Round(float64(xp[xOff+i] * id)))
			op[bOff+4+i] = byte(q)
		}
		for j := 0; j < 16; j++ {
			sum := int16(0)
			for i := 0; i < 16; i++ {
				sum += int16(int8(op[bOff+4+j*16+i]))
			}
			*(*int16)(unsafe.Pointer(&op[bOff+260+j*2])) = sum
		}
	}
}
