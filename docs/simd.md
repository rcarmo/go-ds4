# go-ds4 SIMD Kernels

## Overview

All quantized dot product hot paths use hand-written assembly for amd64 (AVX2+FMA) and arm64 (NEON). The assembly is in Go Plan 9 syntax (`.s` files), compiled by the Go assembler — no external assembler or CGo needed.

## Kernel Inventory

### `ds4/simd/` — Core Vector Operations

| Function | amd64 | arm64 | Description |
|---|---|---|---|
| `Sdot` | AVX2 FMA | NEON FMLA | F32 dot product |
| `Saxpy` | AVX2 FMA | NEON FMLA | y += α×x |
| `RMSNorm` | AVX2 | NEON | In-place RMSNorm with learned scale |
| `RMSNormNoScale` | AVX2 | NEON | RMSNorm without scale weights |
| `VecSiLUMul` | Fast exp | Fast exp | dst = silu(a) × b |

### `ds4/simd/quant_amd64.s` — DS4 Q8_0

| Function | Instructions | Notes |
|---|---|---|
| `DotQ8_0F32` | VCVTPH2PS + VBROADCASTSS + 4×(VPMOVSXBD + VCVTDQ2PS + VMULPS + VFMADD231PS) | DS4 34-byte blocks with F16 scale. 572ns/4096-dim |

Key detail: DS4 Q8_0 uses **F16 scales** (2 bytes + 32 int8 = 34 bytes/block), not the standard ggml F32 scale (36 bytes). The kernel uses `VCVTPH2PS` (F16C extension, available on all AVX2 CPUs) for scale decode.

### `ds4/simd/quant_q2k_amd64.s` — Q2_K

| Function | Instructions | Notes |
|---|---|---|
| `DotQ2Group16` | VPAND + VPSRLW + VPUNPCKLBW + VPMADDUBSW + VPMADDWD | 16 Q2 values from 4 packed bytes. 879ns/4096-dim |

Extracts 2-bit values via shift+mask into XMM, interleaves into sequential order, then signed×signed multiply via the unsigned offset trick.

### `ds4/simd/quant_iq2_amd64.s` — IQ2_XXS

| Function | Instructions | Notes |
|---|---|---|
| `DotIQ2Group32` | VMOVQ×4 + VPUNPCKLQDQ + VINSERTI128 + VPADDB + VPMADDUBSW + VPMADDWD | Zero-copy grid lookup. 807ns/4096-dim |

Loads 4 grid pointers (4×8 bytes) directly into one YMM register without intermediate buffer copy. Uses the unsigned offset trick (+128) for signed×signed dot, with correction via VPMOVSXBW sum.

### `ds4/simd/quant_amd64.s` — DotI8

| Function | Instructions | Notes |
|---|---|---|
| `DotI8` | VPADDB(+128) + VPMADDUBSW + VPMADDWD + VPMOVSXBW(correction) | 6ns/32-elem. Used by Q2K and IQ2 inner loops |

Signed×signed int8 dot via unsigned offset trick: `a_u8 = a_i8 + 128`, then `VPMADDUBSW(a_u8, b_i8)`, with correction `128 × Σb_i8` subtracted.

## The Unsigned Offset Trick

AVX2 `VPMADDUBSW` requires one unsigned and one signed operand. For signed×signed:

```
dot(a_i8, b_i8) = dot(a_i8 + 128, b_i8) - 128 × sum(b_i8)
```

1. Add 128 to `a` (makes it unsigned: range [0, 255])
2. `VPMADDUBSW(a_unsigned, b_signed)` → 16 int16
3. `VPMADDWD(ones, result)` → 8 int32
4. Horizontal sum → raw dot
5. Subtract `128 × sum(b)` correction

This is the standard technique used by llama.cpp, ggml, and all quantized inference engines.

## Fallbacks

- `quant_other.go`: Pure Go scalar fallback for non-amd64/non-arm64
- `quant_arm64.go`: Go scalar with ARM64 F16 conversion (NEON assembly pending)
- `fastexp.go`: Schraudolph fast exp approximation for SiLU (used on all platforms)
