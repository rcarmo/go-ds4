package gpu

import (
	"fmt"
	"unsafe"
)

// Optimized Q2_K GEMV: unrolled 2-bit extraction + warp shuffle reduction.
// Each block = one output row, 256 threads.
// Q2_K block (84 bytes/256 elements): scales[16] + qs[64] + d(f16) + dmin(f16)

const DS4GemvQ2KOptPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry q2k_gemv_opt(
    .param .u64 param_act,
    .param .u64 param_wt,
    .param .u64 param_out,
    .param .u32 param_nBlocks,
    .param .u32 param_outDim,
    .param .u32 param_rowBytes
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<16>;
    .reg .f16 %h0, %h1;
    .reg .pred %p<4>;

    .shared .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_outDim];
    setp.ge.u32 %p0, %r0, %r2;
    @%p0 bra done;

    ld.param.u32 %r3, [param_nBlocks];
    ld.param.u32 %r4, [param_rowBytes];
    ld.param.u64 %rd0, [param_act];
    ld.param.u64 %rd1, [param_wt];
    mul.wide.u32 %rd3, %r0, %r4;
    add.u64 %rd4, %rd1, %rd3;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;

block_loop:
    setp.ge.u32 %p1, %r5, %r3;
    @%p1 bra warp_reduce;

    mul.lo.u32 %r6, %r5, 84;
    cvt.u64.u32 %rd5, %r6;
    add.u64 %rd6, %rd4, %rd5;
    ld.global.b16 %h0, [%rd6+80];
    cvt.f32.f16 %f1, %h0;
    ld.global.b16 %h1, [%rd6+82];
    cvt.f32.f16 %f2, %h1;

    shl.b32 %r7, %r5, 8;
    mul.wide.u32 %rd8, %r7, 4;
    add.u64 %rd9, %rd0, %rd8;

    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.u32 %r8, 0;

group_loop:
    setp.ge.u32 %p2, %r8, 16;
    @%p2 bra group_done;

    cvt.u64.u32 %rd10, %r8;
    add.u64 %rd11, %rd6, %rd10;
    ld.global.u8 %r9, [%rd11];
    and.b32 %r10, %r9, 0xF;
    shr.u32 %r11, %r9, 4;
    cvt.rn.f32.u32 %f5, %r11;

    // Sum activation for bsums: load 4 floats at once
    shl.b32 %r12, %r8, 4;
    mul.wide.u32 %rd12, %r12, 4;
    add.u64 %rd13, %rd9, %rd12;

    // Load 4x f32 activation values (vectorized)
    ld.global.f32 %f6, [%rd13];
    ld.global.f32 %f7, [%rd13+4];
    ld.global.f32 %f8, [%rd13+8];
    ld.global.f32 %f9, [%rd13+12];
    mov.f32 %f10, 0f00000000;
    add.f32 %f10, %f6, %f7;
    add.f32 %f10, %f10, %f8;
    add.f32 %f10, %f10, %f9;
    ld.global.f32 %f6, [%rd13+16];
    ld.global.f32 %f7, [%rd13+20];
    ld.global.f32 %f8, [%rd13+24];
    ld.global.f32 %f9, [%rd13+28];
    add.f32 %f10, %f10, %f6;
    add.f32 %f10, %f10, %f7;
    add.f32 %f10, %f10, %f8;
    add.f32 %f10, %f10, %f9;
    ld.global.f32 %f6, [%rd13+32];
    ld.global.f32 %f7, [%rd13+36];
    ld.global.f32 %f8, [%rd13+40];
    ld.global.f32 %f9, [%rd13+44];
    add.f32 %f10, %f10, %f6;
    add.f32 %f10, %f10, %f7;
    add.f32 %f10, %f10, %f8;
    add.f32 %f10, %f10, %f9;
    ld.global.f32 %f6, [%rd13+48];
    ld.global.f32 %f7, [%rd13+52];
    ld.global.f32 %f8, [%rd13+56];
    ld.global.f32 %f9, [%rd13+60];
    add.f32 %f10, %f10, %f6;
    add.f32 %f10, %f10, %f7;
    add.f32 %f10, %f10, %f8;
    add.f32 %f10, %f10, %f9;
    fma.rn.f32 %f4, %f5, %f10, %f4;

    // Q2 dot: load 4 packed bytes, extract 16 Q2 values, dot with activation
    shl.b32 %r15, %r8, 2;
    add.u32 %r15, %r15, 16;
    cvt.u64.u32 %rd16, %r15;
    add.u64 %rd17, %rd6, %rd16;
    ld.global.u32 %r17, [%rd17];

    mov.f32 %f11, 0f00000000;

    // Byte 0: 4 Q2 values
    and.b32 %r19, %r17, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f6, %f11;
    shr.u32 %r19, %r17, 2;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f7, %f11;
    shr.u32 %r19, %r17, 4;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f8, %f11;
    shr.u32 %r19, %r17, 6;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f9, %f11;

    // Reload act for bytes 1-3
    ld.global.f32 %f6, [%rd13+16];
    ld.global.f32 %f7, [%rd13+20];
    ld.global.f32 %f8, [%rd13+24];
    ld.global.f32 %f9, [%rd13+28];
    shr.u32 %r19, %r17, 8;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f6, %f11;
    shr.u32 %r19, %r17, 10;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f7, %f11;
    shr.u32 %r19, %r17, 12;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f8, %f11;
    shr.u32 %r19, %r17, 14;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f9, %f11;

    ld.global.f32 %f6, [%rd13+32];
    ld.global.f32 %f7, [%rd13+36];
    ld.global.f32 %f8, [%rd13+40];
    ld.global.f32 %f9, [%rd13+44];
    shr.u32 %r19, %r17, 16;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f6, %f11;
    shr.u32 %r19, %r17, 18;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f7, %f11;
    shr.u32 %r19, %r17, 20;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f8, %f11;
    shr.u32 %r19, %r17, 22;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f9, %f11;

    ld.global.f32 %f6, [%rd13+48];
    ld.global.f32 %f7, [%rd13+52];
    ld.global.f32 %f8, [%rd13+56];
    ld.global.f32 %f9, [%rd13+60];
    shr.u32 %r19, %r17, 24;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f6, %f11;
    shr.u32 %r19, %r17, 26;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f7, %f11;
    shr.u32 %r19, %r17, 28;
    and.b32 %r19, %r19, 3;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f8, %f11;
    shr.u32 %r19, %r17, 30;
    cvt.rn.f32.u32 %f12, %r19;
    fma.rn.f32 %f11, %f12, %f9, %f11;

    cvt.rn.f32.u32 %f12, %r10;
    fma.rn.f32 %f3, %f12, %f11, %f3;

    add.u32 %r8, %r8, 1;
    bra group_loop;
group_done:

    mul.f32 %f3, %f1, %f3;
    mul.f32 %f4, %f2, %f4;
    sub.f32 %f3, %f3, %f4;
    add.f32 %f0, %f0, %f3;

    add.u32 %r5, %r5, 256;
    bra block_loop;

warp_reduce:
    .reg .f32 %fs;
    shfl.sync.down.b32 %fs, %f0, 16, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %fs;
    shfl.sync.down.b32 %fs, %f0, 8, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %fs;
    shfl.sync.down.b32 %fs, %f0, 4, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %fs;
    shfl.sync.down.b32 %fs, %f0, 2, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %fs;
    shfl.sync.down.b32 %fs, %f0, 1, 31, 0xFFFFFFFF;
    add.f32 %f0, %f0, %fs;

    and.b32 %r27, %r1, 31;
    setp.ne.u32 %p2, %r27, 0;
    @%p2 bra done;
    shr.u32 %r28, %r1, 5;
    mul.lo.u32 %r29, %r28, 4;
    mov.u64 %rd20, sdata;
    cvt.u64.u32 %rd21, %r29;
    add.u64 %rd22, %rd20, %rd21;
    st.shared.f32 [%rd22], %f0;
    bar.sync 0;
    setp.ne.u32 %p2, %r1, 0;
    @%p2 bra done;
    mov.f32 %f0, 0f00000000;
    mov.u32 %r28, 0;
ws:  setp.ge.u32 %p3, %r28, 8;
@%p3 bra st;
    mul.lo.u32 %r29, %r28, 4;
    cvt.u64.u32 %rd21, %r29;
    add.u64 %rd22, %rd20, %rd21;
    ld.shared.f32 %f8, [%rd22];
    add.f32 %f0, %f0, %f8;
    add.u32 %r28, %r28, 1;
    bra ws;
st:
    ld.param.u64 %rd25, [param_out];
    mul.wide.u32 %rd26, %r0, 4;
    add.u64 %rd27, %rd25, %rd26;
    st.global.f32 [%rd27], %f0;
done:
    ret;
}
`

var cudaGemvQ2KOpt CUfunction
var cudaGemvQ2KOptReady bool

func InitCUDAGemvQ2KOpt() bool {
	if cudaGemvQ2KOptReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4GemvQ2KOptPTX, "q2k_gemv_opt")
	if err != nil {
		fmt.Printf("[gpu] Q2K opt PTX load failed: %v\n", err)
		return false
	}
	cudaGemvQ2KOpt = fn
	cudaGemvQ2KOptReady = true
	fmt.Println("[gpu] DS4 Q2_K optimized GEMV kernel compiled")
	return true
}

func CUDAMatvecQ2KOpt(output, activation *Buffer, weightPtr CUdeviceptr, inDim, outDim, rowBytes int) error {
	if !cudaGemvQ2KOptReady {
		return CUDAMatvecQ2K(output, activation, weightPtr, inDim, outDim, rowBytes)
	}
	EnsureContext()
	nBlocks := uint32(inDim / 256)
	outDimU := uint32(outDim)
	rowBytesU := uint32(rowBytes)
	actPtr := activation.Ptr
	outPtr := output.Ptr
	args := []unsafe.Pointer{
		unsafe.Pointer(&actPtr), unsafe.Pointer(&weightPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&nBlocks), unsafe.Pointer(&outDimU), unsafe.Pointer(&rowBytesU),
	}
	return LaunchKernel(cudaGemvQ2KOpt, uint32(outDim), 1, 1, 256, 1, 1, 256*4, args...)
}
