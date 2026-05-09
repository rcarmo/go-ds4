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
		maxv := float32(0)
		for i := 0; i < 256; i++ {
			v := xp[xOff+i]
			av := v
			if av < 0 {
				av = -av
			}
			if av > amax {
				amax = av
				maxv = v
			}
		}
		if amax == 0 {
			*(*float32)(unsafe.Pointer(&op[bOff])) = 0
			for i := 0; i < 256; i++ {
				op[bOff+4+i] = 0
			}
		} else {
			iscale := -127.0 / maxv
			*(*float32)(unsafe.Pointer(&op[bOff])) = 1.0 / iscale
			for i := 0; i < 256; i++ {
				q := int(math.RoundToEven(float64(xp[xOff+i] * iscale)))
				if q > 127 {
					q = 127
				} else if q < -128 {
					q = -128
				}
				op[bOff+4+i] = byte(int8(q))
			}
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
