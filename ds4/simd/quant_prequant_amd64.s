// quant_prequant_amd64.s — Fused DS4 Q8_0 prequant dot: wide float accumulation.
//
// func DotQ8_0PrequantI8(row, xq, xscale, xsum unsafe.Pointer, nBlocks int) float32
//
// DS4 Q8_0 row: f16 scale (2B) + int8[32] (34B/block).
// Prequantized: xq int8[n], xscale float32[nBlocks], xsum float32[nBlocks] (pre-converted sum).
//
// Per block: accumulate 8 float partials from VPMADDUBSW+VPMADDWD+VCVTDQ2PS,
// scaled by combined_scale = f16(w_scale) * xscale[b].
// Correction: scalar accumulate combined_scale * xsum[b], multiply by 128 at end.
//
#include "textflag.h"

TEXT ·DotQ8_0PrequantI8(SB), NOSPLIT, $0-44
    MOVQ    row+0(FP), SI
    MOVQ    xq+8(FP), DI
    MOVQ    xscale+16(FP), DX
    MOVQ    xsum+24(FP), R8
    MOVQ    nBlocks+32(FP), CX

    VXORPS  Y0, Y0, Y0          // Y0 = 8-wide float accumulator
    VXORPS  X14, X14, X14       // X14 = scalar correction

    VMOVDQU dpi8v2_128<>(SB), Y6
    VMOVDQU dpi8v2_ones<>(SB), Y7

    TESTQ   CX, CX
    JZ      dpi8v2_done

dpi8v2_loop:
    // weight bytes + 128 → unsigned
    VMOVDQU 2(SI), Y1
    VPADDB  Y6, Y1, Y1

    // activation int8
    VMOVDQU (DI), Y2

    // unsigned × signed → 8 int32 partials
    VPMADDUBSW Y2, Y1, Y3
    VPMADDWD Y7, Y3, Y3

    // → 8 float
    VCVTDQ2PS Y3, Y3

    // combined_scale = f16(row_scale) * xscale[b]
    VCVTPH2PS (SI), X4
    VMULSS  (DX), X4, X4        // X4 = combined_scale (scalar)
    VBROADCASTSS X4, Y4         // Y4 = [combined × 8]

    // Y0 += combined × partials
    VFMADD231PS Y4, Y3, Y0

    // correction += combined * xsum[b] (both float32)
    VMULSS  (R8), X4, X5
    VADDSS  X5, X14, X14

    ADDQ    $34, SI
    ADDQ    $32, DI
    ADDQ    $4, DX
    ADDQ    $4, R8
    DECQ    CX
    JNZ     dpi8v2_loop

dpi8v2_done:
    // hsum Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    MOVHLPS X0, X1
    ADDPS   X1, X0
    MOVSS   X0, X1
    SHUFPS  $0x55, X0, X0
    ADDSS   X1, X0

    // result = hsum - 128 * correction
    MOVL    $0x43000000, R9     // 128.0f
    MOVD    R9, X15
    VMULSS  X15, X14, X14
    VSUBSS  X14, X0, X0

    MOVSS   X0, ret+40(FP)
    VZEROUPPER
    RET

DATA dpi8v2_128<>+0x00(SB)/4, $0x80808080
DATA dpi8v2_128<>+0x04(SB)/4, $0x80808080
DATA dpi8v2_128<>+0x08(SB)/4, $0x80808080
DATA dpi8v2_128<>+0x0c(SB)/4, $0x80808080
DATA dpi8v2_128<>+0x10(SB)/4, $0x80808080
DATA dpi8v2_128<>+0x14(SB)/4, $0x80808080
DATA dpi8v2_128<>+0x18(SB)/4, $0x80808080
DATA dpi8v2_128<>+0x1c(SB)/4, $0x80808080
GLOBL dpi8v2_128<>(SB), (RODATA+NOPTR), $32

DATA dpi8v2_ones<>+0x00(SB)/4, $0x00010001
DATA dpi8v2_ones<>+0x04(SB)/4, $0x00010001
DATA dpi8v2_ones<>+0x08(SB)/4, $0x00010001
DATA dpi8v2_ones<>+0x0c(SB)/4, $0x00010001
DATA dpi8v2_ones<>+0x10(SB)/4, $0x00010001
DATA dpi8v2_ones<>+0x14(SB)/4, $0x00010001
DATA dpi8v2_ones<>+0x18(SB)/4, $0x00010001
DATA dpi8v2_ones<>+0x1c(SB)/4, $0x00010001
GLOBL dpi8v2_ones<>(SB), (RODATA+NOPTR), $32
