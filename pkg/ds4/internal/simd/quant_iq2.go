package simd

import "unsafe"

// DotIQ2Group32 computes the dot product of 32 grid values (4 × 8 int8
// from IQ2_XXS grid lookups) against 32 Q8 int8 values.
//
// On amd64: loads 4 grid pointers directly into one YMM, no copy needed.
// Uses VPMADDUBSW+VPMADDWD with unsigned offset correction.
//
//go:noescape
func DotIQ2Group32(grid0, grid1, grid2, grid3 unsafe.Pointer, q8 unsafe.Pointer) int32
