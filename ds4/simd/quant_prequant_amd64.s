// quant_prequant_amd64.s — Fused DS4 Q8_0 prequant dot product kernel.
//
// func DotQ8_0PrequantI8(row unsafe.Pointer, xq unsafe.Pointer, xscale unsafe.Pointer, nBlocks int) float32
//
// DS4 Q8_0 row layout per block: f16 scale (2 bytes) + int8[32] (34 bytes total).
// Activation is prequantized: xq[nBlocks*32] int8, xscale[nBlocks] float32.
//
// Per block: result += f16_to_f32(row_scale) * xscale[b] * DotI8_32(row_qs, xq)
//
// Strategy: unroll 2 blocks per iteration. Use VPMADDUBSW+VPMADDWD for int8 dot,
// VCVTPH2PS for f16 scale decode, single scalar FMA accumulation.
//
#include "textflag.h"

// func DotQ8_0PrequantI8(row unsafe.Pointer, xq unsafe.Pointer, xscale unsafe.Pointer, nBlocks int) float32
TEXT ·DotQ8_0PrequantI8(SB), NOSPLIT, $0-36
    MOVQ    row+0(FP), SI       // SI = Q8_0 row data
    MOVQ    xq+8(FP), DI        // DI = prequantized int8 activation
    MOVQ    xscale+16(FP), DX   // DX = per-block float32 scales
    MOVQ    nBlocks+24(FP), CX  // CX = number of blocks

    VXORPS  X0, X0, X0          // X0 = scalar float accumulator

    VMOVDQU dpi8_128<>(SB), Y6  // Y6 = [128 × 32] for unsigned offset
    VMOVDQU dpi8_ones<>(SB), Y7 // Y7 = [1 × 16] int16 for VPMADDWD

    TESTQ   CX, CX
    JZ      dpi8_done

    // Check if we can do 2-block unroll
    MOVQ    CX, R8
    SHRQ    $1, R8              // R8 = nBlocks/2
    TESTQ   R8, R8
    JZ      dpi8_tail1

dpi8_loop2:
    // Block 0
    VMOVDQU 2(SI), Y1           // row_qs[0:32] (weight int8)
    VMOVDQU (DI), Y2            // xq[0:32] (activation int8)

    // Signed × signed via unsigned offset trick
    VPADDB  Y6, Y1, Y3          // Y3 = weight + 128 (unsigned)
    VPMADDUBSW Y2, Y3, Y4      // unsigned(w+128) × signed(xq) → 16 int16
    VPMADDWD Y7, Y4, Y4        // → 8 int32

    // Horizontal sum block 0
    VEXTRACTI128 $1, Y4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0x4E, X4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0xB1, X4, X5
    VPADDD  X5, X4, X4         // X4[0] = raw_dot_0

    // Correction: 128 * sum(xq[0:32])
    VPMOVSXBW X2, Y8
    VPMADDWD Y7, Y8, Y8
    VEXTRACTI128 $1, Y2, X9
    VPMOVSXBW X9, Y9
    VPMADDWD Y7, Y9, Y9
    VPADDD  Y9, Y8, Y8
    VEXTRACTI128 $1, Y8, X9
    VPADDD  X9, X8, X8
    VPSHUFD $0x4E, X8, X9
    VPADDD  X9, X8, X8
    VPSHUFD $0xB1, X8, X9
    VPADDD  X9, X8, X8         // X8[0] = sum(xq_0)
    VPSLLD  $7, X8, X8
    VPSUBD  X8, X4, X4         // X4[0] = signed dot block 0

    VCVTDQ2PS X4, X4            // dot0 as float
    VCVTPH2PS (SI), X10         // row scale 0 (f16 → f32, lane0)
    MULSS   X10, X4             // × row_scale
    MULSS   (DX), X4            // × xscale[b]
    ADDSS   X4, X0              // accumulate

    // Block 1
    VMOVDQU 36(SI), Y1          // row_qs at offset 34+2=36
    VMOVDQU 32(DI), Y2          // xq[32:64]

    VPADDB  Y6, Y1, Y3
    VPMADDUBSW Y2, Y3, Y4
    VPMADDWD Y7, Y4, Y4

    VEXTRACTI128 $1, Y4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0x4E, X4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0xB1, X4, X5
    VPADDD  X5, X4, X4

    VPMOVSXBW X2, Y8
    VPMADDWD Y7, Y8, Y8
    VEXTRACTI128 $1, Y2, X9
    VPMOVSXBW X9, Y9
    VPMADDWD Y7, Y9, Y9
    VPADDD  Y9, Y8, Y8
    VEXTRACTI128 $1, Y8, X9
    VPADDD  X9, X8, X8
    VPSHUFD $0x4E, X8, X9
    VPADDD  X9, X8, X8
    VPSHUFD $0xB1, X8, X9
    VPADDD  X9, X8, X8
    VPSLLD  $7, X8, X8
    VPSUBD  X8, X4, X4

    VCVTDQ2PS X4, X4
    VCVTPH2PS 34(SI), X10       // row scale 1
    MULSS   X10, X4
    MULSS   4(DX), X4
    ADDSS   X4, X0

    ADDQ    $68, SI             // 2 × 34 bytes
    ADDQ    $64, DI             // 2 × 32 bytes
    ADDQ    $8, DX              // 2 × 4 bytes
    DECQ    R8
    JNZ     dpi8_loop2

    // Check for odd trailing block
    TESTQ   $1, CX
    JZ      dpi8_done

dpi8_tail1:
    VMOVDQU 2(SI), Y1
    VMOVDQU (DI), Y2

    VPADDB  Y6, Y1, Y3
    VPMADDUBSW Y2, Y3, Y4
    VPMADDWD Y7, Y4, Y4

    VEXTRACTI128 $1, Y4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0x4E, X4, X5
    VPADDD  X5, X4, X4
    VPSHUFD $0xB1, X4, X5
    VPADDD  X5, X4, X4

    VPMOVSXBW X2, Y8
    VPMADDWD Y7, Y8, Y8
    VEXTRACTI128 $1, Y2, X9
    VPMOVSXBW X9, Y9
    VPMADDWD Y7, Y9, Y9
    VPADDD  Y9, Y8, Y8
    VEXTRACTI128 $1, Y8, X9
    VPADDD  X9, X8, X8
    VPSHUFD $0x4E, X8, X9
    VPADDD  X9, X8, X8
    VPSHUFD $0xB1, X8, X9
    VPADDD  X9, X8, X8
    VPSLLD  $7, X8, X8
    VPSUBD  X8, X4, X4

    VCVTDQ2PS X4, X4
    VCVTPH2PS (SI), X10
    MULSS   X10, X4
    MULSS   (DX), X4
    ADDSS   X4, X0

dpi8_done:
    MOVSS   X0, ret+32(FP)
    VZEROUPPER
    RET

DATA dpi8_128<>+0x00(SB)/4, $0x80808080
DATA dpi8_128<>+0x04(SB)/4, $0x80808080
DATA dpi8_128<>+0x08(SB)/4, $0x80808080
DATA dpi8_128<>+0x0c(SB)/4, $0x80808080
DATA dpi8_128<>+0x10(SB)/4, $0x80808080
DATA dpi8_128<>+0x14(SB)/4, $0x80808080
DATA dpi8_128<>+0x18(SB)/4, $0x80808080
DATA dpi8_128<>+0x1c(SB)/4, $0x80808080
GLOBL dpi8_128<>(SB), (RODATA+NOPTR), $32

DATA dpi8_ones<>+0x00(SB)/4, $0x00010001
DATA dpi8_ones<>+0x04(SB)/4, $0x00010001
DATA dpi8_ones<>+0x08(SB)/4, $0x00010001
DATA dpi8_ones<>+0x0c(SB)/4, $0x00010001
DATA dpi8_ones<>+0x10(SB)/4, $0x00010001
DATA dpi8_ones<>+0x14(SB)/4, $0x00010001
DATA dpi8_ones<>+0x18(SB)/4, $0x00010001
DATA dpi8_ones<>+0x1c(SB)/4, $0x00010001
GLOBL dpi8_ones<>(SB), (RODATA+NOPTR), $32
