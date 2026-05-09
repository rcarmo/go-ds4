package gpu

import (
	"fmt"
	"unsafe"
)

// Parity Q2_K x Q8_K GEMV. Matches ds4.vecDotQ2KQ8K_scalar / ds4.c traversal.
const DS4GemvQ2KQ8KPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry q2k_q8k_gemv(
    .param .u64 param_q8k,
    .param .u64 param_wt,
    .param .u64 param_out,
    .param .u32 param_nBlocks,
    .param .u32 param_outDim,
    .param .u32 param_rowBytes
) {
    .reg .u32 %r<48>;
    .reg .u64 %rd<36>;
    .reg .s32 %s<16>;
    .reg .f32 %f<12>;
    .reg .f16 %h0, %h1;
    .reg .pred %p<5>;
    .shared .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;    // output row
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_outDim];
    setp.ge.u32 %p0, %r0, %r2;
    @%p0 bra done;

    ld.param.u32 %r3, [param_nBlocks];
    ld.param.u32 %r4, [param_rowBytes];
    ld.param.u64 %rd0, [param_q8k];
    ld.param.u64 %rd1, [param_wt];

    mul.wide.u32 %rd2, %r0, %r4;
    add.u64 %rd3, %rd1, %rd2;   // row base

    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;
block_loop:
    setp.ge.u32 %p1, %r5, %r3;
    @%p1 bra reduce;

    mul.lo.u32 %r6, %r5, 84;
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd3, %rd4;   // x Q2_K block
    mul.lo.u32 %r7, %r5, 292;
    cvt.u64.u32 %rd6, %r7;
    add.u64 %rd7, %rd0, %rd6;   // y Q8_K block

    ld.global.b16 %h0, [%rd5+80];
    cvt.f32.f16 %f1, %h0;       // d
    ld.global.b16 %h1, [%rd5+82];
    cvt.f32.f16 %f2, %h1;       // dmin
    ld.global.f32 %f3, [%rd7];  // yd

    // summs = sum int16(bsums[j]) * int(sc[j] >> 4)
    mov.u32 %r34, 0;
    mov.u32 %r8, 0;
summs_loop:
    setp.ge.u32 %p2, %r8, 16;
    @%p2 bra summs_done;
    cvt.u64.u32 %rd8, %r8;
    add.u64 %rd9, %rd5, %rd8;
    ld.global.u8 %r9, [%rd9];
    shr.u32 %r10, %r9, 4;
    mul.lo.u32 %r11, %r8, 2;
    add.u32 %r11, %r11, 260;    // 4 + 256 + j*2
    cvt.u64.u32 %rd10, %r11;
    add.u64 %rd11, %rd7, %rd10;
    ld.global.s16 %r35, [%rd11];
    mov.b32 %r36, %r10;
    mad.lo.s32 %r34, %r35, %r36, %r34;
    add.u32 %r8, %r8, 1;
    bra summs_loop;
summs_done:

    mov.u32 %r37, 0;             // isum
    mov.u32 %r12, 0;            // is
    mov.u32 %r13, 16;           // q2off (absolute inside block)
    mov.u32 %r14, 4;            // q8off (absolute inside Q8K block)
    mov.u32 %r15, 0;            // k outer
outer_loop:
    setp.ge.u32 %p2, %r15, 2;
    @%p2 bra outer_done;
    mov.u32 %r16, 0;            // shift
    mov.u32 %r17, 0;            // j
inner_loop:
    setp.ge.u32 %p3, %r17, 4;
    @%p3 bra inner_done;

    // first 16: sc[is]&0xf, q2[q2off:q2off+16], q8[q8off:q8off+16]
    cvt.u64.u32 %rd12, %r12;
    add.u64 %rd13, %rd5, %rd12;
    ld.global.u8 %r18, [%rd13];
    and.b32 %r18, %r18, 15;
    mov.u32 %r38, 0;
    mov.u32 %r19, 0;
dot_a:
    setp.ge.u32 %p4, %r19, 16;
    @%p4 bra dot_a_done;
    add.u32 %r20, %r13, %r19;
    cvt.u64.u32 %rd14, %r20;
    add.u64 %rd15, %rd5, %rd14;
    ld.global.u8 %r21, [%rd15];
    shr.u32 %r21, %r21, %r16;
    and.b32 %r21, %r21, 3;
    add.u32 %r22, %r14, %r19;
    cvt.u64.u32 %rd16, %r22;
    add.u64 %rd17, %rd7, %rd16;
    ld.global.s8 %r39, [%rd17];
    mov.b32 %r40, %r21;
    mad.lo.s32 %r38, %r40, %r39, %r38;
    add.u32 %r19, %r19, 1;
    bra dot_a;
dot_a_done:
    mov.b32 %r41, %r18;
    mad.lo.s32 %r37, %r41, %r38, %r37;
    add.u32 %r12, %r12, 1;

    // second 16: sc[is]&0xf, q2[q2off+16:q2off+32], q8[q8off+16:q8off+32]
    cvt.u64.u32 %rd12, %r12;
    add.u64 %rd13, %rd5, %rd12;
    ld.global.u8 %r18, [%rd13];
    and.b32 %r18, %r18, 15;
    mov.u32 %r38, 0;
    mov.u32 %r19, 0;
dot_b:
    setp.ge.u32 %p4, %r19, 16;
    @%p4 bra dot_b_done;
    add.u32 %r20, %r13, 16;
    add.u32 %r20, %r20, %r19;
    cvt.u64.u32 %rd14, %r20;
    add.u64 %rd15, %rd5, %rd14;
    ld.global.u8 %r21, [%rd15];
    shr.u32 %r21, %r21, %r16;
    and.b32 %r21, %r21, 3;
    add.u32 %r22, %r14, 16;
    add.u32 %r22, %r22, %r19;
    cvt.u64.u32 %rd16, %r22;
    add.u64 %rd17, %rd7, %rd16;
    ld.global.s8 %r39, [%rd17];
    mov.b32 %r40, %r21;
    mad.lo.s32 %r38, %r40, %r39, %r38;
    add.u32 %r19, %r19, 1;
    bra dot_b;
dot_b_done:
    mov.b32 %r41, %r18;
    mad.lo.s32 %r37, %r41, %r38, %r37;
    add.u32 %r12, %r12, 1;

    add.u32 %r16, %r16, 2;
    add.u32 %r14, %r14, 32;
    add.u32 %r17, %r17, 1;
    bra inner_loop;
inner_done:
    add.u32 %r13, %r13, 32;
    add.u32 %r15, %r15, 1;
    bra outer_loop;
outer_done:

    cvt.rn.f32.s32 %f4, %r37;
    cvt.rn.f32.s32 %f5, %r34;
    mul.f32 %f6, %f3, %f1;
    mul.f32 %f7, %f3, %f2;
    mul.f32 %f4, %f6, %f4;
    neg.f32 %f11, %f7;
    fma.rn.f32 %f4, %f11, %f5, %f4;
    add.f32 %f0, %f0, %f4;

    add.u32 %r5, %r5, 256;
    bra block_loop;

reduce:
    mov.u64 %rd20, sdata;
    mul.lo.u32 %r30, %r1, 4;
    cvt.u64.u32 %rd21, %r30;
    add.u64 %rd22, %rd20, %rd21;
    st.shared.f32 [%rd22], %f0;
    bar.sync 0;
    mov.u32 %r31, 128;
reduce_loop:
    setp.eq.u32 %p2, %r31, 0;
    @%p2 bra store;
    setp.ge.u32 %p3, %r1, %r31;
    @%p3 bra reduce_skip;
    add.u32 %r32, %r1, %r31;
    mul.lo.u32 %r33, %r32, 4;
    cvt.u64.u32 %rd23, %r33;
    add.u64 %rd24, %rd20, %rd23;
    ld.shared.f32 %f8, [%rd24];
    ld.shared.f32 %f9, [%rd22];
    add.f32 %f9, %f9, %f8;
    st.shared.f32 [%rd22], %f9;
reduce_skip:
    bar.sync 0;
    shr.u32 %r31, %r31, 1;
    bra reduce_loop;
store:
    setp.ne.u32 %p3, %r1, 0;
    @%p3 bra done;
    mov.u64 %rd28, sdata;
    ld.shared.f32 %f10, [%rd28];
    ld.param.u64 %rd25, [param_out];
    mul.wide.u32 %rd26, %r0, 4;
    add.u64 %rd27, %rd25, %rd26;
    st.global.f32 [%rd27], %f10;
done:
    ret;
}
`

var cudaGemvQ2KQ8K CUfunction
var cudaGemvQ2KQ8KReady bool

func CudaGemvQ2KQ8KReady() bool { return cudaGemvQ2KQ8KReady }

func InitCUDAGemvQ2KQ8K() bool {
	if cudaGemvQ2KQ8KReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4GemvQ2KQ8KPTX, "q2k_q8k_gemv")
	if err != nil {
		fmt.Printf("[gpu] Q2KxQ8K PTX load failed: %v\n", err)
		return false
	}
	cudaGemvQ2KQ8K = fn
	cudaGemvQ2KQ8KReady = true
	fmt.Println("[gpu] DS4 Q2_KxQ8_K parity GEMV kernel compiled")
	return true
}

func CUDAMatvecQ2KQ8K(output *Buffer, q8kPtr CUdeviceptr, weightPtr CUdeviceptr, inDim, outDim, rowBytes int, stream CUstream) error {
	if !cudaGemvQ2KQ8KReady {
		return fmt.Errorf("Q2KxQ8K kernel not compiled")
	}
	EnsureContext()
	nBlocks := uint32(inDim / 256)
	outDimU := uint32(outDim)
	rowBytesU := uint32(rowBytes)
	outPtr := output.Ptr
	args := []unsafe.Pointer{
		unsafe.Pointer(&q8kPtr), unsafe.Pointer(&weightPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&nBlocks), unsafe.Pointer(&outDimU), unsafe.Pointer(&rowBytesU),
	}
	if stream != 0 {
		return LaunchKernelStream(cudaGemvQ2KQ8K, uint32(outDim), 1, 1, 256, 1, 1, 256*4, stream, args...)
	}
	return LaunchKernel(cudaGemvQ2KQ8K, uint32(outDim), 1, 1, 256, 1, 1, 256*4, args...)
}

// Parity IQ2_XXS x Q8_K GEMV. Matches ds4.vecDotIQ2XXSQ8K_scalar.
const DS4GemvIQ2Q8KPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry iq2_q8k_gemv(
    .param .u64 param_q8k,
    .param .u64 param_wt,
    .param .u64 param_out,
    .param .u64 param_grid,
    .param .u32 param_nBlocks,
    .param .u32 param_outDim,
    .param .u32 param_rowBytes
) {
    .reg .u32 %r<64>;
    .reg .u64 %rd<40>;
    .reg .f32 %f<12>;
    .reg .f16 %h0;
    .reg .pred %p<6>;
    .shared .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_outDim];
    setp.ge.u32 %p0, %r0, %r2;
    @%p0 bra iq_done;

    ld.param.u32 %r3, [param_nBlocks];
    ld.param.u32 %r4, [param_rowBytes];
    ld.param.u64 %rd0, [param_q8k];
    ld.param.u64 %rd1, [param_wt];
    ld.param.u64 %rd2, [param_grid];
    mul.wide.u32 %rd3, %r0, %r4;
    add.u64 %rd4, %rd1, %rd3;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;
iq_block_loop:
    setp.ge.u32 %p1, %r5, %r3;
    @%p1 bra iq_reduce;

    mul.lo.u32 %r6, %r5, 66;
    cvt.u64.u32 %rd5, %r6;
    add.u64 %rd6, %rd4, %rd5;
    mul.lo.u32 %r7, %r5, 292;
    cvt.u64.u32 %rd7, %r7;
    add.u64 %rd8, %rd0, %rd7;

    ld.global.b16 %h0, [%rd6];
    cvt.f32.f16 %f1, %h0;
    ld.global.f32 %f2, [%rd8];
    mul.f32 %f1, %f1, %f2;
    mul.f32 %f1, %f1, 0f3E000000; // 0.125*d*yd

    mov.u32 %r8, 0;    // bsum in r8, signed b32
    mov.u32 %r9, 0;    // ib32
iq_group_loop:
    setp.ge.u32 %p2, %r9, 8;
    @%p2 bra iq_group_done;

    mul.lo.u32 %r10, %r9, 8;
    add.u32 %r10, %r10, 2;
    cvt.u64.u32 %rd9, %r10;
    add.u64 %rd10, %rd6, %rd9;
    ld.global.u32 %r11, [%rd10];     // grid indices
    ld.global.u32 %r12, [%rd10+4];   // signs + scale
    shr.u32 %r13, %r12, 28;
    shl.b32 %r13, %r13, 1;
    add.u32 %r13, %r13, 1;           // ls
    mov.u32 %r14, 0;                 // sumi
    mov.u32 %r15, 0;                 // l
iq_sub_loop:
    setp.ge.u32 %p3, %r15, 4;
    @%p3 bra iq_sub_done;

    shl.b32 %r16, %r15, 3;
    shr.u32 %r17, %r11, %r16;
    and.b32 %r17, %r17, 255;         // grid idx
    mul.lo.u32 %r18, %r15, 7;
    shr.u32 %r19, %r12, %r18;
    and.b32 %r19, %r19, 127;         // sign idx
    shl.b32 %r20, %r17, 7;
    add.u32 %r20, %r20, %r19;
    shl.b32 %r20, %r20, 3;
    cvt.u64.u32 %rd11, %r20;
    add.u64 %rd12, %rd2, %rd11;

    shl.b32 %r21, %r9, 5;
    shl.b32 %r22, %r15, 3;
    add.u32 %r21, %r21, %r22;
    add.u32 %r21, %r21, 4;           // q8 byte offset
    cvt.u64.u32 %rd13, %r21;
    add.u64 %rd14, %rd8, %rd13;

    mov.u32 %r23, 0;                 // dot
    mov.u32 %r24, 0;
iq_dot8:
    setp.ge.u32 %p4, %r24, 8;
    @%p4 bra iq_dot8_done;
    cvt.u64.u32 %rd15, %r24;
    add.u64 %rd16, %rd12, %rd15;
    add.u64 %rd17, %rd14, %rd15;
    ld.global.s8 %r25, [%rd16];
    ld.global.s8 %r26, [%rd17];
    mad.lo.s32 %r23, %r25, %r26, %r23;
    add.u32 %r24, %r24, 1;
    bra iq_dot8;
iq_dot8_done:
    add.s32 %r14, %r14, %r23;
    add.u32 %r15, %r15, 1;
    bra iq_sub_loop;
iq_sub_done:
    mad.lo.s32 %r8, %r14, %r13, %r8;
    add.u32 %r9, %r9, 1;
    bra iq_group_loop;
iq_group_done:
    cvt.rn.f32.s32 %f3, %r8;
    fma.rn.f32 %f0, %f1, %f3, %f0;
    add.u32 %r5, %r5, 256;
    bra iq_block_loop;

iq_reduce:
    mov.u64 %rd20, sdata;
    mul.lo.u32 %r30, %r1, 4;
    cvt.u64.u32 %rd21, %r30;
    add.u64 %rd22, %rd20, %rd21;
    st.shared.f32 [%rd22], %f0;
    bar.sync 0;
    mov.u32 %r31, 128;
iq_reduce_loop:
    setp.eq.u32 %p2, %r31, 0;
    @%p2 bra iq_store;
    setp.ge.u32 %p3, %r1, %r31;
    @%p3 bra iq_reduce_skip;
    add.u32 %r32, %r1, %r31;
    mul.lo.u32 %r33, %r32, 4;
    cvt.u64.u32 %rd23, %r33;
    add.u64 %rd24, %rd20, %rd23;
    ld.shared.f32 %f4, [%rd24];
    ld.shared.f32 %f5, [%rd22];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd22], %f5;
iq_reduce_skip:
    bar.sync 0;
    shr.u32 %r31, %r31, 1;
    bra iq_reduce_loop;
iq_store:
    setp.ne.u32 %p3, %r1, 0;
    @%p3 bra iq_done;
    mov.u64 %rd25, sdata;
    ld.shared.f32 %f6, [%rd25];
    ld.param.u64 %rd26, [param_out];
    mul.wide.u32 %rd27, %r0, 4;
    add.u64 %rd28, %rd26, %rd27;
    st.global.f32 [%rd28], %f6;
iq_done:
    ret;
}
`

var cudaGemvIQ2Q8K CUfunction
var cudaGemvIQ2Q8KReady bool

func CudaGemvIQ2Q8KReady() bool { return cudaGemvIQ2Q8KReady }

func InitCUDAGemvIQ2Q8K() bool {
	if cudaGemvIQ2Q8KReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4GemvIQ2Q8KPTX, "iq2_q8k_gemv")
	if err != nil {
		fmt.Printf("[gpu] IQ2xQ8K PTX load failed: %v\n", err)
		return false
	}
	cudaGemvIQ2Q8K = fn
	cudaGemvIQ2Q8KReady = true
	fmt.Println("[gpu] DS4 IQ2_XXSxQ8_K parity GEMV kernel compiled")
	return true
}

func CUDAMatvecIQ2Q8K(output *Buffer, q8kPtr CUdeviceptr, weightPtr CUdeviceptr, inDim, outDim, rowBytes int, stream CUstream) error {
	if !cudaGemvIQ2Q8KReady {
		return fmt.Errorf("IQ2xQ8K kernel not compiled")
	}
	EnsureContext()
	nBlocks := uint32(inDim / 256)
	outDimU := uint32(outDim)
	rowBytesU := uint32(rowBytes)
	outPtr := output.Ptr
	gridPtr := cudaGridBuf.Ptr
	args := []unsafe.Pointer{unsafe.Pointer(&q8kPtr), unsafe.Pointer(&weightPtr), unsafe.Pointer(&outPtr), unsafe.Pointer(&gridPtr), unsafe.Pointer(&nBlocks), unsafe.Pointer(&outDimU), unsafe.Pointer(&rowBytesU)}
	if stream != 0 {
		return LaunchKernelStream(cudaGemvIQ2Q8K, uint32(outDim), 1, 1, 256, 1, 1, 256*4, stream, args...)
	}
	return LaunchKernel(cudaGemvIQ2Q8K, uint32(outDim), 1, 1, 256, 1, 1, 256*4, args...)
}
