//go:build !amd64 && !arm64

package simd

import (
	"math"
	"unsafe"
)

// DotQ8_0F32 scalar fallback (DS4 layout: f16 scale + int8[32], 34 bytes).
func DotQ8_0F32(wq8 unsafe.Pointer, x unsafe.Pointer, nBlocks int) float32 {
	wp := (*[1 << 30]byte)(wq8)
	xp := (*[1 << 30]float32)(x)
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		off := b * 34
		scale := f16tof32(*(*uint16)(unsafe.Pointer(&wp[off])))
		xOff := b * 32
		var fsum float32
		for i := 0; i < 32; i++ {
			fsum += float32(int8(wp[off+2+i])) * xp[xOff+i]
		}
		sum += scale * fsum
	}
	return sum
}

// DotF16F32 scalar fallback.
func DotF16F32(wf16 unsafe.Pointer, x unsafe.Pointer, n int) float32 {
	wp := (*[1 << 30]uint16)(wf16)
	xp := (*[1 << 30]float32)(x)
	sum := float32(0)
	for i := 0; i < n; i++ {
		sum += f16tof32(wp[i]) * xp[i]
	}
	return sum
}

func f16tof32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 {
		return math.Float32frombits(sign)
	}
	if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000)
	}
	return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

// QuantizeQ8K scalar fallback.
func QuantizeQ8K(x unsafe.Pointer, out unsafe.Pointer, n int) {
	quantizeQ8KScalar(x, out, n)
}

// DotI8 scalar fallback.
func DotI8(a, b unsafe.Pointer, n int) int32 {
	ap := (*[1 << 30]int8)(a)
	bp := (*[1 << 30]int8)(b)
	sum := int32(0)
	for i := 0; i < n; i++ {
		sum += int32(ap[i]) * int32(bp[i])
	}
	return sum
}

// DotQ8_0PrequantI8 scalar fallback.
func DotQ8_0PrequantI8(row unsafe.Pointer, xq unsafe.Pointer, xscale unsafe.Pointer, xsum unsafe.Pointer, nBlocks int) float32 {
	rp := (*[1 << 30]byte)(row)
	xqp := (*[1 << 30]int8)(xq)
	xsp := (*[1 << 30]float32)(xscale)
	_ = xsum // not needed for scalar path
	sum := float32(0)
	for b := 0; b < nBlocks; b++ {
		off := b * 34
		d := f16tof32(*(*uint16)(unsafe.Pointer(&rp[off])))
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
		// bsums
		for j := 0; j < 16; j++ {
			sum := int16(0)
			for i := 0; i < 16; i++ {
				sum += int16(int8(op[bOff+4+j*16+i]))
			}
			*(*int16)(unsafe.Pointer(&op[bOff+260+j*2])) = sum
		}
	}
}
