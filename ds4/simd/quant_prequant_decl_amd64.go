//go:build amd64

package simd

import "unsafe"

// DotQ8_0PrequantF16 is a legacy amd64 asm variant kept for vet/linker
// parity; current Go callers use DotQ8_0PrequantI8.
//
//go:noescape
func DotQ8_0PrequantF16(row unsafe.Pointer, xq unsafe.Pointer, xscale unsafe.Pointer, nBlocks int) float32
