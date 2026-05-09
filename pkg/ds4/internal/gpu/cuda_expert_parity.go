package gpu

import (
	"fmt"
	"unsafe"
)

const DS4ExpertSwiGLUQ8KPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry expert_swiglu_q8k(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u64 param_weight,
    .param .u64 param_out,
    .param .u32 param_ffnDim,
    .param .u32 param_nExperts,
    .param .f32 param_clamp
) {
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<20>;
    .reg .s32 %si<2>;
    .reg .pred %p<8>;
    .shared .f32 s_abs[256];
    .shared .f32 s_val[256];

    mov.u32 %r0, %ctaid.x;     // block id = expert * blocksPerExpert + q8 block
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_ffnDim];
    ld.param.u32 %r3, [param_nExperts];
    shr.u32 %r4, %r2, 8;       // blocks per expert, ffnDim/256
    div.u32 %r5, %r0, %r4;     // expert
    rem.u32 %r6, %r0, %r4;     // q8 block within expert
    setp.ge.u32 %p0, %r5, %r3;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];
    ld.param.u64 %rd2, [param_weight];
    ld.param.u64 %rd3, [param_out];
    ld.param.f32 %f10, [param_clamp];

    mul.lo.u32 %r7, %r5, %r2;
    shl.b32 %r8, %r6, 8;
    add.u32 %r9, %r7, %r8;     // element base
    add.u32 %r10, %r9, %r1;    // element idx
    mul.wide.u32 %rd4, %r10, 4;
    add.u64 %rd5, %rd0, %rd4;
    add.u64 %rd6, %rd1, %rd4;

    ld.global.f32 %f0, [%rd5]; // gate
    ld.global.f32 %f1, [%rd6]; // up
    setp.gt.f32 %p1, %f10, 0f358637BD; // > 1e-6 approx
    @!%p1 bra clamp_done;
    setp.gt.f32 %p2, %f0, %f10;
    @%p2 mov.f32 %f0, %f10;
    setp.gt.f32 %p3, %f1, %f10;
    @%p3 mov.f32 %f1, %f10;
    neg.f32 %f11, %f10;
    setp.lt.f32 %p4, %f1, %f11;
    @%p4 mov.f32 %f1, %f11;
clamp_done:
    // silu(gate) * up * route_weight. Match CPU fastSigmoid/fastExp bit trick.
    setp.gt.f32 %p5, %f0, 0f41200000; // x > 10
    @%p5 mov.f32 %f3, 0f3F800000;
    @%p5 bra sigmoid_done;
    setp.lt.f32 %p5, %f0, 0fC1200000; // x < -10
    @%p5 mov.f32 %f3, 0f00000000;
    @%p5 bra sigmoid_done;
    neg.f32 %f2, %f0;
    mul.rn.f32 %f2, %f2, 0f4B38AA3B; // (1/ln2) * 2^23
    cvt.rzi.s32.f32 %r13, %f2;
    add.s32 %r13, %r13, 1065353216;
    mov.b32 %f2, %r13;
    add.f32 %f3, %f2, 0f3F800000;
    rcp.rn.f32 %f3, %f3;
sigmoid_done:
    mul.f32 %f4, %f0, %f3;
    mul.f32 %f4, %f4, %f1;
    mul.wide.u32 %rd7, %r5, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.f32 %f5, [%rd8];
    mul.f32 %f4, %f4, %f5;

    abs.f32 %f6, %f4;
    mul.lo.u32 %r11, %r1, 4;
    cvt.u64.u32 %rd9, %r11;
    mov.u64 %rd10, s_abs;
    add.u64 %rd11, %rd10, %rd9;
    st.shared.f32 [%rd11], %f6;
    mov.u64 %rd12, s_val;
    add.u64 %rd13, %rd12, %rd9;
    st.shared.f32 [%rd13], %f4;
    bar.sync 0;

    mov.u32 %r12, 128;
max_loop:
    setp.eq.u32 %p5, %r12, 0;
    @%p5 bra max_done;
    setp.ge.u32 %p6, %r1, %r12;
    @%p6 bra max_skip;
    add.u32 %r13, %r1, %r12;
    mul.lo.u32 %r14, %r13, 4;
    cvt.u64.u32 %rd12, %r14;
    add.u64 %rd13, %rd10, %rd12;
    ld.shared.f32 %f7, [%rd13];
    ld.shared.f32 %f8, [%rd11];
    setp.gt.f32 %p7, %f7, %f8;
    @!%p7 bra keep_max;
    st.shared.f32 [%rd11], %f7;
    mov.u64 %rd14, s_val;
    add.u64 %rd13, %rd14, %rd12;
    ld.shared.f32 %f16, [%rd13];
    add.u64 %rd14, %rd14, %rd9;
    st.shared.f32 [%rd14], %f16;
keep_max:
max_skip:
    bar.sync 0;
    shr.u32 %r12, %r12, 1;
    bra max_loop;
max_done:
    mov.u64 %rd12, s_abs;
    ld.shared.f32 %f9, [%rd12];
    mov.u64 %rd13, s_val;
    ld.shared.f32 %f15, [%rd13];
    setp.neu.f32 %p7, %f15, 0f00000000;
    @%p7 div.rn.f32 %f12, 0fC2FE0000, %f15; // -127/signed_max
    @!%p7 mov.f32 %f12, 0f00000000;
    mul.f32 %f13, %f4, %f12;
    cvt.rni.s32.f32 %r15, %f13;
    max.s32 %r15, %r15, -128;
    min.s32 %r15, %r15, 127;

    // out layout per expert: Q8_K blocks, each 292 bytes: d(f32), qs[256], bsums[16]
    mul.lo.u32 %r16, %r0, 292;
    cvt.u64.u32 %rd14, %r16;
    add.u64 %rd15, %rd3, %rd14;
    setp.ne.u32 %p5, %r1, 0;
    @%p5 bra store_q;
    rcp.rn.f32 %f14, %f12;
    st.global.f32 [%rd15], %f14;
store_q:
    add.u64 %rd16, %rd15, 4;
    cvt.u64.u32 %rd17, %r1;
    add.u64 %rd18, %rd16, %rd17;
    st.global.s8 [%rd18], %r15;
    bar.sync 0;
    // bsums: int16 sum over each 16-lane group, used by Q2_K dmin term.
    and.b32 %r17, %r1, 15;
    setp.ne.u32 %p6, %r17, 0;
    @%p6 bra done;
    shr.u32 %r18, %r1, 4;
    mov.s32 %si0, 0;
    mov.u32 %r19, 0;
bsum_loop:
    setp.ge.u32 %p6, %r19, 16;
    @%p6 bra bsum_done;
    shl.b32 %r12, %r18, 4;
    add.u32 %r12, %r12, %r19;
    add.u32 %r12, %r12, 4;
    cvt.u64.u32 %rd19, %r12;
    add.u64 %rd19, %rd15, %rd19;
    ld.global.s8 %si1, [%rd19];
    add.s32 %si0, %si0, %si1;
    add.u32 %r19, %r19, 1;
    bra bsum_loop;
bsum_done:
    shl.b32 %r19, %r18, 1;
    add.u32 %r19, %r19, 260;
    cvt.u64.u32 %rd19, %r19;
    add.u64 %rd19, %rd15, %rd19;
    st.global.u16 [%rd19], %si0;
done:
    ret;
}
`

var cudaExpertSwiGLUQ8K CUfunction
var cudaExpertSwiGLUQ8KReady bool

func InitCUDAExpertSwiGLUQ8K() bool {
	if cudaExpertSwiGLUQ8KReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4ExpertSwiGLUQ8KPTX, "expert_swiglu_q8k")
	if err != nil {
		fmt.Printf("[gpu] expert SwiGLU Q8K PTX load failed: %v\n", err)
		return false
	}
	cudaExpertSwiGLUQ8K = fn
	cudaExpertSwiGLUQ8KReady = true
	fmt.Println("[gpu] DS4 expert SwiGLU+Q8_K kernel compiled")
	return true
}

func CudaExpertSwiGLUQ8KReady() bool { return cudaExpertSwiGLUQ8KReady }

func CUDAExpertSwiGLUQ8K(gate, up *Buffer, weightsPtr, outQ8K CUdeviceptr, ffnDim, nExperts int, clamp float32, stream CUstream) error {
	if !cudaExpertSwiGLUQ8KReady {
		return fmt.Errorf("expert SwiGLU Q8K kernel not compiled")
	}
	blocksPerExpert := uint32(ffnDim / 256)
	gridX := uint32(nExperts) * blocksPerExpert
	ffnDimU := uint32(ffnDim)
	nExpertsU := uint32(nExperts)
	gatePtr := gate.Ptr
	upPtr := up.Ptr
	args := []unsafe.Pointer{
		unsafe.Pointer(&gatePtr), unsafe.Pointer(&upPtr), unsafe.Pointer(&weightsPtr), unsafe.Pointer(&outQ8K),
		unsafe.Pointer(&ffnDimU), unsafe.Pointer(&nExpertsU), unsafe.Pointer(&clamp),
	}
	return LaunchKernelStream(cudaExpertSwiGLUQ8K, gridX, 1, 1, 256, 1, 1, 256*4, stream, args...)
}
