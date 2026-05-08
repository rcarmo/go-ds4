package simd

import "unsafe"

// DotQ8_0F32 computes dot(Q8_0_weight, F32_activation) for n elements.
// Q8_0 block: float32 scale + int8[32] = 36 bytes per 32 elements.
//
//go:noescape
func DotQ8_0F32(wq8 unsafe.Pointer, x unsafe.Pointer, nBlocks int) float32

// DotF16F32 computes dot(F16_weight, F32_activation) for n elements.
// Uses VCVTPH2PS (F16C) for 8-wide F16→F32 conversion + FMA.
//
//go:noescape
func DotF16F32(wf16 unsafe.Pointer, x unsafe.Pointer, n int) float32

// QuantizeQ8K quantizes n float32 values to Q8_K blocks.
// n must be a multiple of 256 (QK_K).
// out must have room for n/256 * 292 bytes.
//
//go:noescape
func QuantizeQ8K(x unsafe.Pointer, out unsafe.Pointer, n int)

// DotI8 computes the int32 dot product of n int8 values.
// Used as a building block for Q2_K and IQ2_XXS dot products.
//
//go:noescape
func DotI8(a, b unsafe.Pointer, n int) int32

// DotQ8_0PrequantF16 computes one DS4 Q8_0 row dot with prequantized activation.
// Row layout per block: f16 scale + int8[32] values (34 bytes).
// xq layout per block: int8[32], xscale: float32 per block.
// Uses fused AVX2 VPMADDUBSW+VPMADDWD inner loop on amd64.
//
//go:noescape
func DotQ8_0PrequantI8(row unsafe.Pointer, xq unsafe.Pointer, xscale unsafe.Pointer, nBlocks int) float32
