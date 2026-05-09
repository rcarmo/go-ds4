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

    // Load scale byte
    cvt.u64.u32 %rd10, %r8;
    add.u64 %rd11, %rd6, %rd10;
    ld.global.u8 %r9, [%rd11];
    and.b32 %r10, %r9, 0xF;
    shr.u32 %r11, %r9, 4;
    cvt.rn.f32.u32 %f5, %r11;

    // Activation base for this group: act[j*16]
    shl.b32 %r12, %r8, 4;
    mul.wide.u32 %rd12, %r12, 4;
    add.u64 %rd13, %rd9, %rd12;

    // Bsums: sum 16 activation values (scalar, simple, correct)
    mov.f32 %f10, 0f00000000;
    mov.u32 %r13, 0;
bsum_loop:
    setp.ge.u32 %p3, %r13, 16;
    @%p3 bra bsum_done;
    shl.b32 %r14, %r13, 2;
    cvt.u64.u32 %rd14, %r14;
    add.u64 %rd15, %rd13, %rd14;
    ld.global.f32 %f6, [%rd15];
    add.f32 %f10, %f10, %f6;
    add.u32 %r13, %r13, 1;
    bra bsum_loop;
bsum_done:
    fma.rn.f32 %f4, %f5, %f10, %f4;

    // Q2 dot: load 4 packed bytes as u32, extract 16 Q2 values
    shl.b32 %r15, %r8, 2;
    add.u32 %r15, %r15, 16;
    cvt.u64.u32 %rd16, %r15;
    add.u64 %rd17, %rd6, %rd16;
    ld.global.u32 %r17, [%rd17];

    mov.f32 %f11, 0f00000000;

    // 16 Q2 values x activation: byte-by-byte extraction, reload act each time
    mov.u32 %r18, 0;
q2_loop:
    setp.ge.u32 %p3, %r18, 16;
    @%p3 bra q2_done;

    // Extract Q2 value: (packed >> (k*2)) & 3
    shl.b32 %r19, %r18, 1;
    shr.u32 %r20, %r17, %r19;
    and.b32 %r20, %r20, 3;
    cvt.rn.f32.u32 %f12, %r20;

    // Load activation[j*16 + k]
    shl.b32 %r21, %r18, 2;
    cvt.u64.u32 %rd18, %r21;
    add.u64 %rd19, %rd13, %rd18;
    ld.global.f32 %f13, [%rd19];

    fma.rn.f32 %f11, %f12, %f13, %f11;
    add.u32 %r18, %r18, 1;
    bra q2_loop;
q2_done:

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
warp_sum:
    setp.ge.u32 %p3, %r28, 8;
    @%p3 bra store;
    mul.lo.u32 %r29, %r28, 4;
    cvt.u64.u32 %rd21, %r29;
    add.u64 %rd22, %rd20, %rd21;
    ld.shared.f32 %f8, [%rd22];
    add.f32 %f0, %f0, %f8;
    add.u32 %r28, %r28, 1;
    bra warp_sum;
store:
    ld.param.u64 %rd23, [param_out];
    mul.wide.u32 %rd21, %r0, 4;
    add.u64 %rd22, %rd23, %rd21;
    st.global.f32 [%rd22], %f0;
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

func CUDAMatvecQ2KOptStream(output, activation *Buffer, weightPtr CUdeviceptr, inDim, outDim, rowBytes int, stream CUstream) error {
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
	return LaunchKernelStream(cudaGemvQ2KOpt, uint32(outDim), 1, 1, 256, 1, 1, 256*4, stream, args...)
}
