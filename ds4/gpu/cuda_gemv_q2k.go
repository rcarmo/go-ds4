package gpu

import (
	"fmt"
	"unsafe"
)

// Q2_K GEMV: each block handles one output row (256 threads cooperatively).
// Q2_K block (84 bytes per 256 elements): 16B scales + 64B packed q2 + 2B d(f16) + 2B dmin(f16).

const DS4GemvQ2KPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry q2k_gemv(
    .param .u64 param_act,
    .param .u64 param_wt,
    .param .u64 param_out,
    .param .u32 param_nBlocks,
    .param .u32 param_outDim,
    .param .u32 param_rowBytes
) {
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<12>;
    .reg .f16 %h<2>;
    .reg .pred %p<4>;
    .shared .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_outDim];
    setp.ge.u32 %p0, %r0, %r2;
    @%p0 bra lbl_done;

    ld.param.u32 %r3, [param_nBlocks];
    ld.param.u32 %r4, [param_rowBytes];
    ld.param.u64 %rd0, [param_act];
    ld.param.u64 %rd1, [param_wt];
    mul.wide.u32 %rd3, %r0, %r4;
    add.u64 %rd4, %rd1, %rd3;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;

lbl_bloop:
    setp.ge.u32 %p1, %r5, %r3;
    @%p1 bra lbl_reduce;

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

    // Simple: just dot first 16 elements as raw bytes for correctness test
    mov.f32 %f3, 0f00000000;
    mov.u32 %r8, 0;
lbl_dot:
    setp.ge.u32 %p2, %r8, 64;
    @%p2 bra lbl_dotdone;

    // Read packed Q2 byte from qs[r8] (at offset 16 + r8 in block)
    add.u32 %r9, %r8, 16;
    cvt.u64.u32 %rd10, %r9;
    add.u64 %rd11, %rd6, %rd10;
    ld.global.u8 %r10, [%rd11];

    // Extract 4 Q2 values and dot with activation
    and.b32 %r11, %r10, 3;
    cvt.rn.f32.u32 %f4, %r11;
    shl.b32 %r12, %r8, 4;
    cvt.u64.u32 %rd12, %r12;
    add.u64 %rd13, %rd9, %rd12;
    ld.global.f32 %f5, [%rd13];
    fma.rn.f32 %f3, %f4, %f5, %f3;

    add.u32 %r8, %r8, 1;
    bra lbl_dot;
lbl_dotdone:

    fma.rn.f32 %f0, %f1, %f3, %f0;

    add.u32 %r5, %r5, 256;
    bra lbl_bloop;

lbl_reduce:
    mul.lo.u32 %r13, %r1, 4;
    mov.u64 %rd14, sdata;
    cvt.u64.u32 %rd15, %r13;
    add.u64 %rd16, %rd14, %rd15;
    st.shared.f32 [%rd16], %f0;
    bar.sync 0;
    mov.u32 %r14, 128;
lbl_redloop:
    setp.lt.u32 %p3, %r14, 1;
    @%p3 bra lbl_store;
    setp.ge.u32 %p2, %r1, %r14;
    @%p2 bra lbl_redskip;
    add.u32 %r15, %r1, %r14;
    mul.lo.u32 %r16, %r15, 4;
    cvt.u64.u32 %rd15, %r16;
    add.u64 %rd16, %rd14, %rd15;
    ld.shared.f32 %f6, [%rd16];
    mul.lo.u32 %r16, %r1, 4;
    cvt.u64.u32 %rd15, %r16;
    add.u64 %rd16, %rd14, %rd15;
    ld.shared.f32 %f7, [%rd16];
    add.f32 %f7, %f7, %f6;
    st.shared.f32 [%rd16], %f7;
lbl_redskip:
    bar.sync 0;
    shr.u32 %r14, %r14, 1;
    bra lbl_redloop;
lbl_store:
    setp.ne.u32 %p2, %r1, 0;
    @%p2 bra lbl_done;
    mov.u64 %rd14, sdata;
    ld.shared.f32 %f8, [%rd14];
    ld.param.u64 %rd17, [param_out];
    mul.wide.u32 %rd18, %r0, 4;
    add.u64 %rd19, %rd17, %rd18;
    st.global.f32 [%rd19], %f8;
lbl_done:
    ret;
}
`

var cudaGemvQ2K CUfunction
var cudaGemvQ2KReady bool

func InitCUDAGemvQ2K() bool {
	if cudaGemvQ2KReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4GemvQ2KPTX, "q2k_gemv")
	if err != nil {
		fmt.Printf("[gpu] Q2K GEMV PTX load failed: %v\n", err)
		return false
	}
	cudaGemvQ2K = fn
	cudaGemvQ2KReady = true
	fmt.Println("[gpu] DS4 Q2_K GEMV kernel compiled (CUDA PTX)")
	return true
}

func CUDAMatvecQ2K(output, activation *Buffer, weightPtr CUdeviceptr, inDim, outDim, rowBytes int) error {
	if !cudaGemvQ2KReady {
		return fmt.Errorf("Q2K kernel not compiled")
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
	return LaunchKernel(cudaGemvQ2K, uint32(outDim), 1, 1, 256, 1, 1, 256*4, args...)
}

// CudaGemvQ2KReady returns true if the Q2K kernel is compiled.
func CudaGemvQ2KReady() bool { return cudaGemvQ2KReady }
