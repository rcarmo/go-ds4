//go:build amd64

package simd

import "unsafe"

// DotQ2x32 computes the dot product of 128 Q2 values (32 packed bytes)
// against 128 Q8 int8 values using AVX2 VPMADDUBSW.
// Q2 values are extracted via VPAND+VPSRLW into 4 groups of 32.
//
//go:noescape
func DotQ2x32(q2 unsafe.Pointer, q8 unsafe.Pointer) int32
