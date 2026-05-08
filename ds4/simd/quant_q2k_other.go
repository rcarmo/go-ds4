//go:build !amd64

package simd

import "unsafe"

// DotQ2Group16 scalar fallback.
func DotQ2Group16(qs unsafe.Pointer, q8 unsafe.Pointer) int32 {
	qp := (*[4]byte)(qs)
	q8p := (*[16]int8)(q8)
	sum := int32(0)
	for k := 0; k < 4; k++ {
		b := qp[k]
		sum += int32(b&3) * int32(q8p[k*4])
		sum += int32((b>>2)&3) * int32(q8p[k*4+1])
		sum += int32((b>>4)&3) * int32(q8p[k*4+2])
		sum += int32(b>>6) * int32(q8p[k*4+3])
	}
	return sum
}
