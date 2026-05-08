//go:build arm64

package simd

import (
	"math"
	"unsafe"
)

func f16tof32Arm(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
	}
	if exp == 31 {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000)
	}
	return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

// quantizeQ8KScalar is the Go scalar implementation called by the arm64 asm stub.
// DotQ8_0PrequantF16 scalar fallback on arm64 (DS4 layout: 34-byte blocks).
func DotQ8_0PrequantF16(row unsafe.Pointer, xq unsafe.Pointer, xscale unsafe.Pointer, nBlocks int) float32 {
	rp := (*[1 << 30]byte)(row)
	xqp := (*[1 << 30]int8)(xq)
	xsp := (*[1 << 30]float32)(xscale)
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		off := b * 34
		d := f16tof32Arm(*(*uint16)(unsafe.Pointer(&rp[off])))
		dot := int32(0)
		for i := 0; i < 32; i++ {
			dot += int32(int8(rp[off+2+i])) * int32(xqp[b*32+i])
		}
		sum += d * xsp[b] * float32(dot)
	}
	return sum
}

func quantizeQ8KScalar(x unsafe.Pointer, out unsafe.Pointer, n int) {
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
