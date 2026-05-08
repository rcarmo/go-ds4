// quant_avx2_amd64.s — Full-block AVX2 kernels for Q2_K and IQ2_XXS dot products.
//
// Q2_K: extract 2-bit values via VPAND+VPSRLW, VPMADDUBSW with Q8 values.
// Processes 32 Q2 bytes (128 Q2 values) per iteration.
//
// IQ2_XXS: grid lookup + VPMADDUBSW on expanded signed grid values.
// Processes 32 grid values per SDOT-equivalent operation.
//
#include "textflag.h"

// func DotQ2x32(q2 *byte, q8 *byte) int32
//
// Computes dot product of 128 Q2 values (32 packed bytes) against 128 Q8 int8 values.
// Q2 values are unsigned [0,3], Q8 values are signed int8.
// Uses VPMADDUBSW (Q2 as unsigned, Q8 as signed).
//
// 128 values = 4 groups of 32:
//   Group 0: q2[i] & 3      for i in 0..31  → 32 bytes
//   Group 1: (q2[i]>>2) & 3 for i in 0..31  → 32 bytes
//   Group 2: (q2[i]>>4) & 3 for i in 0..31  → 32 bytes
//   Group 3: (q2[i]>>6) & 3 for i in 0..31  → 32 bytes
// Each group dotted with q8[g*32 : (g+1)*32]
TEXT ·DotQ2x32(SB), NOSPLIT, $0-20
    MOVQ    q2+0(FP), SI
    MOVQ    q8+8(FP), DI

    // Load 32 packed Q2 bytes
    VMOVDQU (SI), Y0             // Y0 = raw packed bytes

    // Extract 4 groups of 32 Q2 values
    VPBROADCASTB dq2x_mask<>(SB), Y6  // Y6 = [0x03 × 32]
    VMOVDQU dq2x_ones<>(SB), Y7      // Y7 = [1 × 16] int16 for VPMADDWD

    // Group 0: bits 0-1
    VPAND   Y6, Y0, Y1          // Y1 = q2 & 3 (32 unsigned bytes)
    VMOVDQU (DI), Y2             // Y2 = q8[0:32] (signed)
    VPMADDUBSW Y2, Y1, Y3       // Y3 = 16 int16 partial dots
    VPMADDWD Y7, Y3, Y3         // Y3 = 8 int32

    // Group 1: bits 2-3
    VPSRLW  $2, Y0, Y1
    VPAND   Y6, Y1, Y1
    VMOVDQU 32(DI), Y2
    VPMADDUBSW Y2, Y1, Y4
    VPMADDWD Y7, Y4, Y4
    VPADDD  Y4, Y3, Y3          // accumulate

    // Group 2: bits 4-5
    VPSRLW  $4, Y0, Y1
    VPAND   Y6, Y1, Y1
    VMOVDQU 64(DI), Y2
    VPMADDUBSW Y2, Y1, Y4
    VPMADDWD Y7, Y4, Y4
    VPADDD  Y4, Y3, Y3

    // Group 3: bits 6-7
    VPSRLW  $6, Y0, Y1
    VPAND   Y6, Y1, Y1
    VMOVDQU 96(DI), Y2
    VPMADDUBSW Y2, Y1, Y4
    VPMADDWD Y7, Y4, Y4
    VPADDD  Y4, Y3, Y3

    // Horizontal sum Y3 → scalar int32
    VEXTRACTI128 $1, Y3, X4
    VPADDD  X4, X3, X3
    VPSHUFD $0x4E, X3, X4
    VPADDD  X4, X3, X3
    VPSHUFD $0xB1, X3, X4
    VPADDD  X4, X3, X3

    MOVL    X3, ret+16(FP)
    VZEROUPPER
    RET

DATA dq2x_mask<>+0(SB)/1, $0x03
GLOBL dq2x_mask<>(SB), (RODATA+NOPTR), $1

DATA dq2x_ones<>+0x00(SB)/4, $0x00010001
DATA dq2x_ones<>+0x04(SB)/4, $0x00010001
DATA dq2x_ones<>+0x08(SB)/4, $0x00010001
DATA dq2x_ones<>+0x0c(SB)/4, $0x00010001
DATA dq2x_ones<>+0x10(SB)/4, $0x00010001
DATA dq2x_ones<>+0x14(SB)/4, $0x00010001
DATA dq2x_ones<>+0x18(SB)/4, $0x00010001
DATA dq2x_ones<>+0x1c(SB)/4, $0x00010001
GLOBL dq2x_ones<>(SB), (RODATA+NOPTR), $32
