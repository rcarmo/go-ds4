// quant_iq2_amd64.s — AVX2 IQ2_XXS dot product kernel
//
// func DotIQ2Group32(grid0, grid1, grid2, grid3 *byte, q8 *byte) int32
//
// Computes dot product of 32 grid values (4 × 8 signed int8 from grid lookups)
// against 32 Q8 signed int8 values. No intermediate buffer copy needed.
//
// Loads 4 grid pointers directly, packs into one YMM, then VPMADDUBSW+VPMADDWD.
//
#include "textflag.h"

// func DotIQ2Group32(grid0, grid1, grid2, grid3 *byte, q8 *byte) int32
TEXT ·DotIQ2Group32(SB), NOSPLIT, $0-44
    MOVQ    grid0+0(FP), SI
    MOVQ    grid1+8(FP), DI
    MOVQ    grid2+16(FP), R8
    MOVQ    grid3+24(FP), R9
    MOVQ    q8+32(FP), R10

    // Load 4 × 8 bytes from grid pointers into one YMM register
    // grid0[0:8] → low 64 bits of lane 0
    // grid1[0:8] → high 64 bits of lane 0
    // grid2[0:8] → low 64 bits of lane 1
    // grid3[0:8] → high 64 bits of lane 1
    VMOVQ   (SI), X0             // X0 = [grid0, 0]
    VMOVQ   (DI), X1             // X1 = [grid1, 0]
    VPUNPCKLQDQ X1, X0, X0      // X0 = [grid0, grid1] = 16 bytes

    VMOVQ   (R8), X2             // X2 = [grid2, 0]
    VMOVQ   (R9), X3             // X3 = [grid3, 0]
    VPUNPCKLQDQ X3, X2, X2      // X2 = [grid2, grid3] = 16 bytes

    VINSERTI128 $1, X2, Y0, Y0  // Y0 = [grid0, grid1, grid2, grid3] = 32 bytes

    // Grid values are SIGNED int8. For VPMADDUBSW we need unsigned × signed.
    // Q8 values are signed. Grid values are signed (range depends on codebook).
    // We need signed × signed. Use the offset trick:
    // grid_u8 = grid_i8 + 128
    // dot(grid_i8, q8_i8) = dot(grid_u8, q8_i8) - 128 * sum(q8_i8)

    // Add 128 to make grid unsigned
    VPADDB  diq2_128<>(SB), Y0, Y1  // Y1 = grid + 128 (unsigned)

    // Load 32 Q8 values
    VMOVDQU (R10), Y2            // Y2 = q8[0:32] (signed)

    // VPMADDUBSW: unsigned(Y1) × signed(Y2) → 16 int16
    VPMADDUBSW Y2, Y1, Y3
    // VPMADDWD with ones → 8 int32
    VPMADDWD diq2_ones<>(SB), Y3, Y3

    // Horizontal sum Y3 → scalar
    VEXTRACTI128 $1, Y3, X4
    VPADDD  X4, X3, X3
    VPSHUFD $0x4E, X3, X4
    VPADDD  X4, X3, X3
    VPSHUFD $0xB1, X3, X4
    VPADDD  X4, X3, X3          // X3[0] = raw_dot

    // Correction: subtract 128 * sum(q8)
    // sum(q8) via sign-extend to int16 then sum
    VPMOVSXBW X2, Y4            // low 16 bytes → 16 int16
    VEXTRACTI128 $1, Y2, X5
    VPMOVSXBW X5, Y5            // high 16 bytes → 16 int16
    VPADDW  Y5, Y4, Y4          // 16 int16
    VPMADDWD diq2_ones<>(SB), Y4, Y4
    VEXTRACTI128 $1, Y4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0x4E, X4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0xB1, X4, X5
    VPADDD  X5, X4, X4          // X4[0] = sum(q8)

    // correction = 128 * sum(q8)
    VPSLLD  $7, X4, X4
    VPSUBD  X4, X3, X3          // result = raw - correction

    MOVL    X3, ret+40(FP)
    VZEROUPPER
    RET

DATA diq2_128<>+0x00(SB)/4, $0x80808080
DATA diq2_128<>+0x04(SB)/4, $0x80808080
DATA diq2_128<>+0x08(SB)/4, $0x80808080
DATA diq2_128<>+0x0c(SB)/4, $0x80808080
DATA diq2_128<>+0x10(SB)/4, $0x80808080
DATA diq2_128<>+0x14(SB)/4, $0x80808080
DATA diq2_128<>+0x18(SB)/4, $0x80808080
DATA diq2_128<>+0x1c(SB)/4, $0x80808080
GLOBL diq2_128<>(SB), (RODATA+NOPTR), $32

DATA diq2_ones<>+0x00(SB)/4, $0x00010001
DATA diq2_ones<>+0x04(SB)/4, $0x00010001
DATA diq2_ones<>+0x08(SB)/4, $0x00010001
DATA diq2_ones<>+0x0c(SB)/4, $0x00010001
DATA diq2_ones<>+0x10(SB)/4, $0x00010001
DATA diq2_ones<>+0x14(SB)/4, $0x00010001
DATA diq2_ones<>+0x18(SB)/4, $0x00010001
DATA diq2_ones<>+0x1c(SB)/4, $0x00010001
GLOBL diq2_ones<>(SB), (RODATA+NOPTR), $32
