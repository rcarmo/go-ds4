// quant_amd64.s — AVX2+FMA quantized dot product kernels for DS4.
//
// DotQ8_0F32: Q8_0 weight × F32 activation
// DotF16F32:  F16 weight × F32 activation (VCVTPH2PS, F16C extension)
// QuantizeQ8K: F32 → Q8_K quantization
// DotI8: int8 × int8 → int32 dot product
//
#include "textflag.h"

// func DotQ8_0F32(wq8 unsafe.Pointer, x unsafe.Pointer, nBlocks int) float32
//
// Per block (36 bytes = 4 scale + 32 int8):
//   VBROADCASTSS scale
//   4× (VPMOVSXBD 8 int8→8 int32, VCVTDQ2PS, VMULPS scale, VFMADD231PS with x)
//   = 32 elements per block
TEXT ·DotQ8_0F32(SB), NOSPLIT, $0-28
    MOVQ    wq8+0(FP), SI       // SI = Q8_0 blocks
    MOVQ    x+8(FP), DI         // DI = float32 activation
    MOVQ    nBlocks+16(FP), CX  // CX = number of blocks

    VXORPS  Y0, Y0, Y0          // Y0 = accumulator

    TESTQ   CX, CX
    JZ      q8f32_done

q8f32_block:
    // Broadcast scale
    VBROADCASTSS (SI), Y1       // Y1 = [scale × 8]

    // Process 32 int8 values in 4 groups of 8
    // Group 0: bytes 4-11
    VPMOVSXBD 4(SI), Y2         // sign-extend 8 int8 → 8 int32
    VCVTDQ2PS Y2, Y2            // int32 → float32
    VMULPS  Y1, Y2, Y2          // × scale
    VFMADD231PS (DI), Y2, Y0    // Y0 += Y2 * x[0:8]

    // Group 1: bytes 12-19
    VPMOVSXBD 12(SI), Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y1, Y2, Y2
    VFMADD231PS 32(DI), Y2, Y0

    // Group 2: bytes 20-27
    VPMOVSXBD 20(SI), Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y1, Y2, Y2
    VFMADD231PS 64(DI), Y2, Y0

    // Group 3: bytes 28-35
    VPMOVSXBD 28(SI), Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y1, Y2, Y2
    VFMADD231PS 96(DI), Y2, Y0

    ADDQ    $36, SI              // next Q8_0 block
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
