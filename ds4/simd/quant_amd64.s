// quant_amd64.s — AVX2+FMA quantized dot product kernels for DS4.
//
// DotQ8_0F32: DS4 Q8_0 (f16 scale + 32 int8, 34 bytes) × F32 activation
// DotF16F32:  F16 weight × F32 activation (VCVTPH2PS, F16C extension)
// QuantizeQ8K: F32 → Q8_K quantization
// DotI8: int8 × int8 → int32 dot product
//
#include "textflag.h"

// func DotQ8_0F32(wq8 unsafe.Pointer, x unsafe.Pointer, nBlocks int) float32
//
// Per block (34 bytes = 2-byte f16 scale + 32 int8):
//   VCVTPH2PS load scale (lane0), VBROADCASTSS scale
//   4× (VPMOVSXBD 8 int8→8 int32, VCVTDQ2PS, VMULPS scale, VFMADD231PS with x)
TEXT ·DotQ8_0F32(SB), NOSPLIT, $0-28
    MOVQ    wq8+0(FP), SI       // SI = Q8_0 blocks
    MOVQ    x+8(FP), DI         // DI = float32 activation
    MOVQ    nBlocks+16(FP), CX  // CX = number of blocks

    VXORPS  Y0, Y0, Y0          // Y0 = accumulator

    TESTQ   CX, CX
    JZ      q8f32_done

q8f32_block:
    // Load f16 scale and broadcast
    VCVTPH2PS (SI), X1          // lane0 = scale
    VBROADCASTSS X1, Y1         // Y1 = [scale × 8]

    // Process 32 int8 values in 4 groups of 8
    // Group 0: bytes 2-9
    VPMOVSXBD 2(SI), Y2         // sign-extend 8 int8 → 8 int32
    VCVTDQ2PS Y2, Y2            // int32 → float32
    VMULPS  Y1, Y2, Y2          // × scale
    VFMADD231PS (DI), Y2, Y0    // Y0 += Y2 * x[0:8]

    // Group 1: bytes 10-17
    VPMOVSXBD 10(SI), Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y1, Y2, Y2
    VFMADD231PS 32(DI), Y2, Y0

    // Group 2: bytes 18-25
    VPMOVSXBD 18(SI), Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y1, Y2, Y2
    VFMADD231PS 64(DI), Y2, Y0

    // Group 3: bytes 26-33
    VPMOVSXBD 26(SI), Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y1, Y2, Y2
    VFMADD231PS 96(DI), Y2, Y0

    ADDQ    $34, SI              // next Q8_0 block
    ADDQ    $128, DI             // next 32 floats
    DECQ    CX
    JNZ     q8f32_block

q8f32_done:
    // Horizontal sum Y0 → scalar
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    MOVHLPS X0, X1
    ADDPS   X1, X0
    MOVSS   X0, X1
    SHUFPS  $0x55, X0, X0
    ADDSS   X1, X0

    MOVSS   X0, ret+24(FP)
    VZEROUPPER
    RET

// func DotF16F32(wf16 unsafe.Pointer, x unsafe.Pointer, n int) float32
//
// Uses VCVTPH2PS (F16C, available on all AVX2 CPUs):
//   Converts 8 FP16 → 8 FP32 per instruction.
//   Then VFMADD231PS with activation.
TEXT ·DotF16F32(SB), NOSPLIT, $0-28
    MOVQ    wf16+0(FP), SI      // SI = F16 weights
    MOVQ    x+8(FP), DI         // DI = float32 activation
    MOVQ    n+16(FP), CX        // CX = element count

    VXORPS  Y0, Y0, Y0          // Y0 = accumulator
    VXORPS  Y1, Y1, Y1          // Y1 = second accumulator

    // Main loop: 16 elements per iteration (2 × 8)
    SHRQ    $4, CX              // CX = n/16
    TESTQ   CX, CX
    JZ      f16f32_tail

f16f32_loop:
    // Load 8 FP16, convert to FP32
    VCVTPH2PS (SI), Y2
    VFMADD231PS (DI), Y2, Y0    // Y0 += convert(f16[0:8]) * x[0:8]

    VCVTPH2PS 16(SI), Y3
    VFMADD231PS 32(DI), Y3, Y1  // Y1 += convert(f16[8:16]) * x[8:16]

    ADDQ    $32, SI              // 16 × 2 bytes
    ADDQ    $64, DI              // 16 × 4 bytes
    DECQ    CX
    JNZ     f16f32_loop

f16f32_tail:
    VADDPS  Y1, Y0, Y0          // combine accumulators

    // Horizontal sum
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    MOVHLPS X0, X1
    ADDPS   X1, X0
    MOVSS   X0, X1
    SHUFPS  $0x55, X0, X0
    ADDSS   X1, X0

    MOVSS   X0, ret+24(FP)
    VZEROUPPER
    RET

// func QuantizeQ8K(x unsafe.Pointer, out unsafe.Pointer, n int)
// Delegates to Go scalar for correctness. The hot path is dot products, not quantization.
TEXT ·QuantizeQ8K(SB), NOSPLIT, $0-24
    JMP     ·quantizeQ8KGo(SB)


// func DotQ8_0PrequantF16(row, xq, xscale unsafe.Pointer, nBlocks int) float32
//
// DS4 Q8_0 layout per block: f16 scale + int8[32] (34 bytes).
// Computes Σ_b f16(scale_b) * xscale[b] * dot_i8_32(row_qs[b], xq[b]).
TEXT ·DotQ8_0PrequantF16(SB), NOSPLIT, $0-36
    MOVQ    row+0(FP), SI
    MOVQ    xq+8(FP), DI
    MOVQ    xscale+16(FP), DX
    MOVQ    nBlocks+24(FP), CX

    VXORPS  X0, X0, X0          // scalar accumulator

    TESTQ   CX, CX
    JZ      dq8pf16_done

    VMOVDQU di8_128<>(SB), Y4   // [128 x 32]
    VMOVDQU di8_ones<>(SB), Y5  // int16 ones

dq8pf16_loop:
    // scale = f16_to_f32(*(uint16*)row)
    // VCVTPH2PS m64 -> xmm converts 4 halfs; lane0 is our scale.
    VCVTPH2PS (SI), X1

    // dot_i8_32(row[2:34], xq[0:32]) using signed*signed trick.
    VMOVDQU 2(SI), Y2
    VMOVDQU (DI), Y3

    VPADDB  Y4, Y2, Y6
    VPMADDUBSW Y3, Y6, Y7
    VPMADDWD Y5, Y7, Y7

    // horizontal sum raw dot: Y7 -> X8[0]
    VEXTRACTI128 $1, Y7, X8
    VPADDD  X8, X7, X7
    VPSHUFD $0x4E, X7, X8
    VPADDD  X8, X7, X7
    VPSHUFD $0xB1, X7, X8
    VPADDD  X8, X7, X8          // X8[0] = raw

    // correction = 128 * sum(xq)
    VPMOVSXBW X3, Y9
    VPMADDWD Y5, Y9, Y9
    VEXTRACTI128 $1, Y3, X10
    VPMOVSXBW X10, Y10
    VPMADDWD Y5, Y10, Y10
    VPADDD  Y10, Y9, Y9

    VEXTRACTI128 $1, Y9, X11
    VPADDD  X11, X9, X9
    VPSHUFD $0x4E, X9, X11
    VPADDD  X11, X9, X9
    VPSHUFD $0xB1, X9, X11
    VPADDD  X11, X9, X9         // X9[0] = sum(xq)

    VPSLLD  $7, X9, X9
    VPSUBD  X9, X8, X8          // X8[0] = signed dot

    VCVTDQ2PS X8, X8            // dot -> float32
    MULSS   X1, X8              // * row scale
    MULSS   (DX), X8            // * activation scale
    ADDSS   X8, X0

    ADDQ    $34, SI
    ADDQ    $32, DI
    ADDQ    $4, DX
    DECQ    CX
    JNZ     dq8pf16_loop

dq8pf16_done:
    MOVSS   X0, ret+32(FP)
    VZEROUPPER
    RET

// func DotI8(a, b unsafe.Pointer, n int) int32
//
// Computes Σ int8(a[i]) × int8(b[i]) using VPMADDUBSW + VPMADDWD.
// Used for Q2_K and IQ2_XXS inner loops.
TEXT ·DotI8(SB), NOSPLIT, $0-28
    MOVQ    a+0(FP), SI
    MOVQ    b+8(FP), DI
    MOVQ    n+16(FP), CX

    VXORPS  Y0, Y0, Y0          // Y0 = int32 accumulator

    // For signed × signed, we use the offset trick:
    // a_u8 = a_i8 + 128 (unsigned)
    // VPMADDUBSW(a_u8, b_i8) = Σ a_u8*b_i8
    // correction = 128 * Σ b_i8
    // result = VPMADDUBSW_result - correction

    VMOVDQU di8_128<>(SB), Y4   // Y4 = [128 × 32]
    VMOVDQU di8_ones<>(SB), Y5  // Y5 = [1 × 16] int16 for VPMADDWD

    VXORPS  Y6, Y6, Y6          // Y6 = correction accumulator

    // Main loop: 32 bytes per iteration
    SHRQ    $5, CX              // CX = n/32
    TESTQ   CX, CX
    JZ      di8_done

di8_loop:
    VMOVDQU (SI), Y1            // a[0:32] as int8
    VMOVDQU (DI), Y2            // b[0:32] as int8

    // Offset a to unsigned
    VPADDB  Y4, Y1, Y3          // Y3 = a + 128 (unsigned)

    // VPMADDUBSW: unsigned(Y3) × signed(Y2) → 16 int16
    VPMADDUBSW Y2, Y3, Y7
    // VPMADDWD: 16 int16 × ones → 8 int32
    VPMADDWD Y5, Y7, Y7
    // Accumulate
    VPADDD  Y7, Y0, Y0

    // Correction: sum of b_i8
    VPMOVSXBW X2, Y7            // low 16 bytes → 16 int16
    VPMADDWD Y5, Y7, Y7
    VPADDD  Y7, Y6, Y6
    VEXTRACTI128 $1, Y2, X8
    VPMOVSXBW X8, Y7
    VPMADDWD Y5, Y7, Y7
    VPADDD  Y7, Y6, Y6

    ADDQ    $32, SI
    ADDQ    $32, DI
    DECQ    CX
    JNZ     di8_loop

di8_done:
    // Horizontal sum Y0
    VEXTRACTI128 $1, Y0, X1
    VPADDD  X1, X0, X0
    VPSHUFD $0x4E, X0, X1
    VPADDD  X1, X0, X0
    VPSHUFD $0xB1, X0, X1
    VPADDD  X1, X0, X0          // X0[0] = raw dot

    // Horizontal sum Y6 (correction)
    VEXTRACTI128 $1, Y6, X1
    VPADDD  X1, X6, X6
    VPSHUFD $0x4E, X6, X1
    VPADDD  X1, X6, X6
    VPSHUFD $0xB1, X6, X1
    VPADDD  X1, X6, X6          // X6[0] = sum(b)

    // correction = 128 * sum(b)
    VPSLLD  $7, X6, X6
    VPSUBD  X6, X0, X0          // result = raw - correction

    MOVL    X0, ret+24(FP)
    VZEROUPPER
    RET

DATA di8_128<>+0x00(SB)/4, $0x80808080
DATA di8_128<>+0x04(SB)/4, $0x80808080
DATA di8_128<>+0x08(SB)/4, $0x80808080
DATA di8_128<>+0x0c(SB)/4, $0x80808080
DATA di8_128<>+0x10(SB)/4, $0x80808080
DATA di8_128<>+0x14(SB)/4, $0x80808080
DATA di8_128<>+0x18(SB)/4, $0x80808080
DATA di8_128<>+0x1c(SB)/4, $0x80808080
GLOBL di8_128<>(SB), (RODATA+NOPTR), $32

DATA di8_ones<>+0x00(SB)/4, $0x00010001
DATA di8_ones<>+0x04(SB)/4, $0x00010001
DATA di8_ones<>+0x08(SB)/4, $0x00010001
DATA di8_ones<>+0x0c(SB)/4, $0x00010001
DATA di8_ones<>+0x10(SB)/4, $0x00010001
DATA di8_ones<>+0x14(SB)/4, $0x00010001
DATA di8_ones<>+0x18(SB)/4, $0x00010001
DATA di8_ones<>+0x1c(SB)/4, $0x00010001
GLOBL di8_ones<>(SB), (RODATA+NOPTR), $32
