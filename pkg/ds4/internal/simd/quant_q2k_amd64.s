// quant_q2k_amd64.s — AVX2 Q2_K per-group dot product
//
// func DotQ2Group16(qs *byte, q8 *byte) int32
//
// Computes dot product of 16 Q2 values (4 packed bytes) against 16 Q8 int8 values.
// Extracts 2-bit values, converts to bytes, then integer multiply-accumulate.
//
// 4 packed bytes → 16 values via shift+mask:
//   byte[0]: vals 0-3 (shifts 0,2,4,6)
//   byte[1]: vals 4-7
//   byte[2]: vals 8-11
//   byte[3]: vals 12-15
//
#include "textflag.h"

// func DotQ2Group16(qs *byte, q8 *byte) int32
TEXT ·DotQ2Group16(SB), NOSPLIT, $0-20
    MOVQ    qs+0(FP), SI
    MOVQ    q8+8(FP), DI

    // Load 4 Q2 bytes and 16 Q8 bytes into XMM
    MOVL    (SI), AX

    // Expand 4 bytes → 16 Q2 values in XMM register
    // Strategy: broadcast the dword, use 4 different shift+mask combinations
    VMOVD   AX, X0
    VPBROADCASTD X0, X0          // X0 = [packed × 4]

    // Shift each dword copy by 0,2,4,6 then mask with 0x03
    // But all 4 copies need DIFFERENT shifts → need VPSRLVD (AVX2)
    // Or: separate shifts
    VPAND   dq2g_m03<>(SB), X0, X1   // bits 0-1: 4 values (in each dword pos 0)
    VPSRLW  $2, X0, X2
    VPAND   dq2g_m03<>(SB), X2, X2   // bits 2-3
    VPSRLW  $4, X0, X3
    VPAND   dq2g_m03<>(SB), X3, X3   // bits 4-5
    VPSRLW  $6, X0, X4
    VPAND   dq2g_m03<>(SB), X4, X4   // bits 6-7

    // X1 has Q2 vals at positions {0,4,8,12} in each 32-bit lane
    // X2 at {1,5,9,13}, X3 at {2,6,10,14}, X4 at {3,7,11,15}
    // We need to interleave them back to sequential order
    // Use VPUNPCKLBW to interleave pairs:
    VPUNPCKLBW X2, X1, X5       // X5 = [v0,v1, v4,v5, v8,v9, v12,v13, ...]
    VPUNPCKLBW X4, X3, X6       // X6 = [v2,v3, v6,v7, v10,v11, v14,v15, ...]
    VPUNPCKLWD X6, X5, X7       // X7 = [v0,v1,v2,v3, v4,v5,v6,v7, ...] sequential!

    // X7 now has 16 Q2 values (unsigned bytes 0-3) in sequential order
    // Load 16 Q8 signed bytes
    VMOVDQU (DI), X8

    // VPMADDUBSW: unsigned(X7) × signed(X8) → 8 int16
    VPMADDUBSW X8, X7, X9
    // VPMADDWD with ones → 4 int32
    VPMADDWD dq2g_ones<>(SB), X9, X9

    // Horizontal sum 4 int32
    VPSHUFD $0x4E, X9, X10
    VPADDD  X10, X9, X9
    VPSHUFD $0xB1, X9, X10
    VPADDD  X10, X9, X9

    MOVL    X9, ret+16(FP)
    VZEROUPPER
    RET

DATA dq2g_m03<>+0x00(SB)/4, $0x03030303
DATA dq2g_m03<>+0x04(SB)/4, $0x03030303
DATA dq2g_m03<>+0x08(SB)/4, $0x03030303
DATA dq2g_m03<>+0x0c(SB)/4, $0x03030303
GLOBL dq2g_m03<>(SB), (RODATA+NOPTR), $16

DATA dq2g_ones<>+0x00(SB)/4, $0x00010001
DATA dq2g_ones<>+0x04(SB)/4, $0x00010001
DATA dq2g_ones<>+0x08(SB)/4, $0x00010001
DATA dq2g_ones<>+0x0c(SB)/4, $0x00010001
GLOBL dq2g_ones<>(SB), (RODATA+NOPTR), $16
