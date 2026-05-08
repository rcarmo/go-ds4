//go:build !amd64

package simd

import "unsafe"

// DotQ2x32 scalar fallback.
func DotQ2x32(q2 unsafe.Pointer, q8 unsafe.Pointer) int32 {
	q2p := (*[32]byte)(q2)
	q8p := (*[128]int8)(q8)
	sum := int32(0)
	for i := 0; i < 32; i++ {
		b := q2p[i]
		sum += int32(b&3) * int32(q8p[i*4])
		sum += int32((b>>2)&3) * int32(q8p[i*4+1])
		sum += int32((b>>4)&3) * int32(q8p[i*4+2])
		sum += int32(b>>6) * int32(q8p[i*4+3])
	}
	return sum
}
