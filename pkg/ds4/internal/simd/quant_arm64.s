// quant_arm64.s — NEON quantized dot product kernels for DS4.
//
// DotQ8_0F32: Q8_0 weight × F32 activation
// DotF16F32:  F16 weight × F32 activation (FCVT)
// QuantizeQ8K: F32 → Q8_K quantization
// DotI8: int8 × int8 → int32 dot product (SDOT when available)
//
#include "textflag.h"

// SDOT Vd.4S, Vn.16B, Vm.16B = 0x4E809400 | (Vm<<16) | (Vn<<5) | Vd
#define SDOT_4S(vm, vn, vd) WORD $(0x4E809400 | ((vm)<<16) | ((vn)<<5) | (vd))

// FCVTL Vd.4S, Vn.4H (FP16→FP32, lower 4)
#define FCVTL_4S(vn, vd) WORD $(0x0E217800 | ((vn)<<5) | (vd))
// FCVTL2 Vd.4S, Vn.8H (FP16→FP32, upper 4)
#define FCVTL2_4S(vn, vd) WORD $(0x4E217800 | ((vn)<<5) | (vd))

// SXTL Vd.8H, Vn.8B (sign-extend int8→int16, lower)
#define SXTL_8B_8H(vn, vd) WORD $(0x0F08A400 | ((vn)<<5) | (vd))
// SXTL2 Vd.8H, Vn.16B (sign-extend int8→int16, upper)
#define SXTL2_16B_8H(vn, vd) WORD $(0x4F08A400 | ((vn)<<5) | (vd))
// SXTL Vd.4S, Vn.4H (sign-extend int16→int32, lower)
#define SXTL_4H_4S(vn, vd) WORD $(0x0F10A400 | ((vn)<<5) | (vd))
// SXTL2 Vd.4S, Vn.8H (sign-extend int16→int32, upper)
#define SXTL2_8H_4S(vn, vd) WORD $(0x4F10A400 | ((vn)<<5) | (vd))

// SCVTF Vd.4S, Vn.4S (signed int32 → float32)
#define SCVTF_4S(vn, vd) WORD $(0x4E21D800 | ((vn)<<5) | (vd))

// FMUL Vd.4S, Vn.4S, Vm.4S
#define FMUL_4S(vm, vn, vd) WORD $(0x6E20DC00 | ((vm)<<16) | ((vn)<<5) | (vd))

// SADDLP Vd.8H, Vn.16B (signed pairwise add long)
#define SADDLP_16B_8H(vn, vd) WORD $(0x4E202800 | ((vn)<<5) | (vd))
// SADDLP Vd.4S, Vn.8H
#define SADDLP_8H_4S(vn, vd) WORD $(0x4E602800 | ((vn)<<5) | (vd))
// ADDV Sd, Vn.4S (add across vector)
#define ADDV_4S(vn, vd) WORD $(0x4EB1B800 | ((vn)<<5) | (vd))

// FABS Vd.4S, Vn.4S
#define FABS_4S(vn, vd) WORD $(0x4EA0F800 | ((vn)<<5) | (vd))
// FMAX Vd.4S, Vn.4S, Vm.4S
#define FMAX_4S(vm, vn, vd) WORD $(0x4E20F400 | ((vm)<<16) | ((vn)<<5) | (vd))
// FMAXV Sd, Vn.4S
#define FMAXV_4S(vn, vd) WORD $(0x6E30F800 | ((vn)<<5) | (vd))
// FCVTNS Vd.4S, Vn.4S (float→int32, round nearest)
#define FCVTNS_4S(vn, vd) WORD $(0x4E21A800 | ((vn)<<5) | (vd))
// SQXTN Vd.8B, Vn.8H (saturating narrow signed)
#define SQXTN_8B(vn, vd) WORD $(0x0E214800 | ((vn)<<5) | (vd))
// SQXTN2 Vd.16B, Vn.8H
#define SQXTN2_16B(vn, vd) WORD $(0x4E214800 | ((vn)<<5) | (vd))
// SQXTN Vd.4H, Vn.4S
#define SQXTN_4H(vn, vd) WORD $(0x0E614800 | ((vn)<<5) | (vd))
// SQXTN2 Vd.8H, Vn.4S
#define SQXTN2_8H(vn, vd) WORD $(0x4E614800 | ((vn)<<5) | (vd))

// func DotQ8_0F32(wq8 unsafe.Pointer, x unsafe.Pointer, nBlocks int) float32
//
// Per Q8_0 block (34 bytes): scale(f16) + int8[32]
// Dequant: SXTL int8→int16→int32, SCVTF→float32, FMUL scale, FMLA with x
TEXT ·DotQ8_0F32(SB), NOSPLIT, $0-28
    MOVD    wq8+0(FP), R0
    MOVD    x+8(FP), R1
    MOVD    nBlocks+16(FP), R2

    VEOR    V0.B16, V0.B16, V0.B16  // accumulator 0
    VEOR    V1.B16, V1.B16, V1.B16  // accumulator 1

    CBZ     R2, q8f32_done

q8f32_blk:
    // Load and broadcast f16 scale.
    VLD1    (R0), [V30.H8]
    FCVTL_4S(30, 30)
    VDUP    V30.S[0], V30.S4

    // Load 32 int8 from R0+2, process in 4 groups of 8
    ADD     $2, R0, R3

    // Group 0: int8[0:8] → int16[0:8] → int32[0:4] and [4:8]
    VLD1    (R3), [V2.B16]          // load 16 bytes
    SXTL_8B_8H(2, 3)                // V3.8H = sext(V2.B8) lower 8
    SXTL_4H_4S(3, 4)                // V4.4S = sext(V3.H4) lower 4
    SCVTF_4S(4, 4)                   // float32
    FMUL_4S(30, 4, 4)               // × scale
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V0.S4

    SXTL2_8H_4S(3, 4)               // V4.4S = sext(V3.H4) upper 4
    SCVTF_4S(4, 4)
    FMUL_4S(30, 4, 4)
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V1.S4

    // Group 1: int8[8:16]
    SXTL2_16B_8H(2, 3)              // V3.8H = sext(V2.B8) upper 8
    SXTL_4H_4S(3, 4)
    SCVTF_4S(4, 4)
    FMUL_4S(30, 4, 4)
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V0.S4

    SXTL2_8H_4S(3, 4)
    SCVTF_4S(4, 4)
    FMUL_4S(30, 4, 4)
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V1.S4

    // Group 2-3: int8[16:32]
    ADD     $16, R3, R3
    VLD1    (R3), [V2.B16]
    SXTL_8B_8H(2, 3)
    SXTL_4H_4S(3, 4)
    SCVTF_4S(4, 4)
    FMUL_4S(30, 4, 4)
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V0.S4

    SXTL2_8H_4S(3, 4)
    SCVTF_4S(4, 4)
    FMUL_4S(30, 4, 4)
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V1.S4

    SXTL2_16B_8H(2, 3)
    SXTL_4H_4S(3, 4)
    SCVTF_4S(4, 4)
    FMUL_4S(30, 4, 4)
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V0.S4

    SXTL2_8H_4S(3, 4)
    SCVTF_4S(4, 4)
    FMUL_4S(30, 4, 4)
    VLD1.P  16(R1), [V5.S4]
    VFMLA   V4.S4, V5.S4, V1.S4

    ADD     $34, R0, R0          // next block
    SUB     $1, R2, R2
    CBNZ    R2, q8f32_blk

q8f32_done:
    // Horizontal sum V0 + V1
    WORD    $(0x4E21D400)        // FADD V0.4S, V0.4S, V1.4S
    VMOV    V0.S[0], R3
    VMOV    V0.S[1], R4
    VMOV    V0.S[2], R5
    VMOV    V0.S[3], R6
    FMOVS   R3, F10
    FMOVS   R4, F11
    FMOVS   R5, F12
    FMOVS   R6, F13
    FADDS   F11, F10, F10
    FADDS   F12, F10, F10
    FADDS   F13, F10, F10

    FMOVS   F10, ret+24(FP)
    RET

// func DotF16F32(wf16 unsafe.Pointer, x unsafe.Pointer, n int) float32
//
// Uses FCVTL/FCVTL2: 4 FP16 → 4 FP32 per instruction.
// 8 elements per iteration (2 × FCVTL + 2 × FMLA).
TEXT ·DotF16F32(SB), NOSPLIT, $0-28
    MOVD    wf16+0(FP), R0
    MOVD    x+8(FP), R1
    MOVD    n+16(FP), R2

    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16

    LSR     $3, R2, R3           // R3 = n/8
    CBZ     R3, f16_done

f16_loop:
    // Load 8 FP16 values (16 bytes)
    VLD1    (R0), [V2.H8]
    ADD     $16, R0, R0

    // Convert lower 4 FP16 → 4 FP32
    FCVTL_4S(2, 3)              // V3.4S = cvt(V2.4H lower)
    VLD1.P  16(R1), [V4.S4]
    VFMLA   V3.S4, V4.S4, V0.S4

    // Convert upper 4 FP16 → 4 FP32
    FCVTL2_4S(2, 3)             // V3.4S = cvt(V2.4H upper)
    VLD1.P  16(R1), [V4.S4]
    VFMLA   V3.S4, V4.S4, V1.S4

    SUB     $1, R3, R3
    CBNZ    R3, f16_loop

f16_done:
    WORD    $(0x4E21D400)        // FADD V0.4S, V0.4S, V1.4S
    VMOV    V0.S[0], R3
    VMOV    V0.S[1], R4
    VMOV    V0.S[2], R5
    VMOV    V0.S[3], R6
    FMOVS   R3, F10
    FMOVS   R4, F11
    FMOVS   R5, F12
    FMOVS   R6, F13
    FADDS   F11, F10, F10
    FADDS   F12, F10, F10
    FADDS   F13, F10, F10

    FMOVS   F10, ret+24(FP)
    RET

// func DotI8(a, b unsafe.Pointer, n int) int32
//
// Uses SDOT (signed dot product, ARMv8.2+):
// 16 int8 × int8 → 4 int32 per SDOT instruction.
TEXT ·DotI8(SB), NOSPLIT, $0-28
    MOVD    a+0(FP), R0
    MOVD    b+8(FP), R1
    MOVD    n+16(FP), R2

    VEOR    V0.B16, V0.B16, V0.B16  // accumulator

    LSR     $4, R2, R3           // R3 = n/16
    CBZ     R3, di8_done

di8_loop:
    VLD1.P  16(R0), [V1.B16]
    VLD1.P  16(R1), [V2.B16]
    SDOT_4S(2, 1, 0)            // V0.4S += sdot(V1.16B, V2.16B)
    SUB     $1, R3, R3
    CBNZ    R3, di8_loop

di8_done:
    ADDV_4S(0, 0)               // V0.S[0] = sum of 4 lanes
    VMOV    V0.S[0], R3
    MOVW    R3, ret+24(FP)
    RET

// func QuantizeQ8K(x unsafe.Pointer, out unsafe.Pointer, n int)
// Scalar fallback for ARM64 (NEON version would need many WORD macros).
// The hot path is the dot products, not quantization.
TEXT ·QuantizeQ8K(SB), NOSPLIT, $0-24
    // For now, delegate to the Go scalar implementation.
    // The quantization is called once per token per layer, not in the inner loop.
    // The Go compiler will generate decent code for the scalar version.
    B       ·quantizeQ8KScalar(SB)
