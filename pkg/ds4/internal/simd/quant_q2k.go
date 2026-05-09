//go:build amd64

package simd

import "unsafe"

// DotQ2Group16 computes the dot product of 16 Q2 values (4 packed bytes)
// against 16 Q8 int8 values using AVX2.
// Returns int32 dot product (Q2 values 0-3, Q8 values signed int8).
//
//go:noescape
func DotQ2Group16(qs unsafe.Pointer, q8 unsafe.Pointer) int32
