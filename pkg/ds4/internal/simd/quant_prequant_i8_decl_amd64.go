//go:build amd64

package simd

import "unsafe"

// DotQ8_0PrequantI8 computes one DS4 Q8_0 row dot with prequantized activation.
// Row layout per block: f16 scale + int8[32] values (34 bytes).
// xq: int8[nBlocks*32], xscale: float32[nBlocks], xsum: float32[nBlocks].
//
//go:noescape
func DotQ8_0PrequantI8(row unsafe.Pointer, xq unsafe.Pointer, xscale unsafe.Pointer, xsum unsafe.Pointer, nBlocks int) float32
