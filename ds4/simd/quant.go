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
