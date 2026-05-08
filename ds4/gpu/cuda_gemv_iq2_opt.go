package gpu

import (
	"fmt"
	"unsafe"
)

// Optimized IQ2_XXS GEMV: vectorized loads + warp shuffle reduction.
// Each block = one output row, 256 threads.
// Per block (66 bytes / 256 elements): f16 d + 8 groups of 8 bytes.
// Each group: 4 grid indices + packed sign/scale aux32.
// Grid lookup: 8 int8 values per (gridIdx, signIdx) pair.
//
// Key optimizations vs naive kernel:
// - Load grid entries as 2x u32 (8 bytes) instead of 8x s8
// - Warp shuffle reduction instead of shared memory tree
// - Fewer branches in inner loop

const DS4GemvIQ2OptPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry iq2xxs_gemv_opt(
    .param .u64 param_act,
    .param .u64 param_wt,
    .param .u64 param_out,
    .param .u64 param_grid,
    .param .u32 param_nBlocks,
    .param .u32 param_outDim,
    .param .u32 param_rowBytes
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<32>;
    .reg .f32 %f<16>;
    .reg .f16 %h0;
    .reg .pred %p<4>;
    .reg .s32 %si;

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
    ld.param.u64 %rd2, [param_grid];

    mul.wide.u32 %rd3, %r0, %r4;
    add.u64 %rd4, %rd1, %rd3;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;

block_loop:
    setp.ge.u32 %p1, %r5, %r3;
    @%p1 bra warp_reduce;

    mul.lo.u32 %r6, %r5, 66;
    cvt.u64.u32 %rd5, %r6;
    add.u64 %rd6, %rd4, %rd5;
    ld.global.b16 %h0, [%rd6];
    cvt.f32.f16 %f1, %h0;

    add.u64 %rd7, %rd6, 2;
    shl.b32 %r7, %r5, 8;
    mul.wide.u32 %rd8, %r7, 4;
    add.u64 %rd9, %rd0, %rd8;

    mov.f32 %f2, 0f00000000;
    mov.u32 %r8, 0;

sg_loop:
    setp.ge.u32 %p2, %r8, 8;
    @%p2 bra sg_done;

    mul.lo.u32 %r9, %r8, 8;
    cvt.u64.u32 %rd10, %r9;
    add.u64 %rd11, %rd7, %rd10;
    ld.global.u32 %r10, [%rd11];
    ld.global.u32 %r11, [%rd11+4];

    shr.u32 %r12, %r11, 28;
    shl.b32 %r12, %r12, 1;
    add.u32 %r12, %r12, 1;
    cvt.rn.f32.s32 %f3, %r12;

    shl.b32 %r16, %r8, 5;
    mul.wide.u32 %rd14, %r16, 4;
    add.u64 %rd15, %rd9, %rd14;

    // Process 4 sub-groups of 8, load grid as 2x u32 for speed
    mov.u32 %r20, 0;
sub_loop:
    setp.ge.u32 %p3, %r20, 4;
    @%p3 bra sub_done;

    shl.b32 %r21, %r20, 3;
    shr.u32 %r13, %r10, %r21;
    and.b32 %r13, %r13, 0xFF;

    mul.lo.u32 %r22, %r20, 7;
    shr.u32 %r14, %r11, %r22;
    and.b32 %r14, %r14, 0x7F;

    shl.b32 %r15, %r13, 7;
    add.u32 %r15, %r15, %r14;
    shl.b32 %r15, %r15, 3;
    cvt.u64.u32 %rd12, %r15;
    add.u64 %rd13, %rd2, %rd12;

    shl.b32 %r23, %r20, 5;
    cvt.u64.u32 %rd16, %r23;
    add.u64 %rd17, %rd15, %rd16;

    // Load grid 8 bytes as 2x u32, extract signed bytes, dot with activation
    ld.global.v2.u32 {%r24, %r25}, [%rd13];

    // Byte 0-3 from %r24
    and.b32 %r26, %r24, 0xFF;
    setp.gt.u32 %p3, %r26, 127;
    @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    shr.u32 %r26, %r24, 8; and.b32 %r26, %r26, 0xFF;
    setp.gt.u32 %p3, %r26, 127; @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17+4];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    shr.u32 %r26, %r24, 16; and.b32 %r26, %r26, 0xFF;
    setp.gt.u32 %p3, %r26, 127; @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17+8];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    shr.u32 %r26, %r24, 24;
    setp.gt.u32 %p3, %r26, 127; @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17+12];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    // Byte 4-7 from %r25
    and.b32 %r26, %r25, 0xFF;
    setp.gt.u32 %p3, %r26, 127; @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17+16];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    shr.u32 %r26, %r25, 8; and.b32 %r26, %r26, 0xFF;
    setp.gt.u32 %p3, %r26, 127; @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17+20];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    shr.u32 %r26, %r25, 16; and.b32 %r26, %r26, 0xFF;
    setp.gt.u32 %p3, %r26, 127; @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17+24];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    shr.u32 %r26, %r25, 24;
    setp.gt.u32 %p3, %r26, 127; @%p3 add.s32 %r26, %r26, -256;
    cvt.rn.f32.s32 %f4, %r26;
    ld.global.f32 %f5, [%rd17+28];
    fma.rn.f32 %f2, %f4, %f5, %f2;

    add.u32 %r20, %r20, 1;
    bra sub_loop;
sub_done:

    add.u32 %r8, %r8, 1;
    bra sg_loop;
sg_done:

    mov.f32 %f7, 0f3E000000;
    mul.f32 %f2, %f1, %f2;
    fma.rn.f32 %f0, %f7, %f2, %f0;

    add.u32 %r5, %r5, 256;
    bra block_loop;

warp_reduce:
    // Warp shuffle reduction (no shared memory needed for intra-warp)
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

    // Lane 0 of each warp writes to shared memory
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

    // Thread 0 sums warp results (8 warps for 256 threads)
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
    ld.param.u64 %rd25, [param_out];
    mul.wide.u32 %rd26, %r0, 4;
    add.u64 %rd27, %rd25, %rd26;
    st.global.f32 [%rd27], %f0;

done:
    ret;
}
`

var cudaGemvIQ2Opt CUfunction
var cudaGemvIQ2OptReady bool

func InitCUDAGemvIQ2Opt() bool {
	if cudaGemvIQ2OptReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4GemvIQ2OptPTX, "iq2xxs_gemv_opt")
	if err != nil {
		fmt.Printf("[gpu] IQ2 opt PTX load failed: %v\n", err)
		return false
	}
	cudaGemvIQ2Opt = fn
	cudaGemvIQ2OptReady = true
	fmt.Println("[gpu] DS4 IQ2_XXS optimized GEMV kernel compiled")
	return true
}

// CUDAMatvecIQ2Opt dispatches the optimized IQ2 GEMV.
func CUDAMatvecIQ2Opt(output, activation *Buffer, weightPtr CUdeviceptr, inDim, outDim, rowBytes int) error {
	if !cudaGemvIQ2OptReady {
		return CUDAMatvecIQ2(output, activation, weightPtr, inDim, outDim, rowBytes)
	}
	EnsureContext()
	nBlocks := uint32(inDim / 256)
	outDimU := uint32(outDim)
	rowBytesU := uint32(rowBytes)
	actPtr := activation.Ptr
	outPtr := output.Ptr
	gridPtr := cudaGridBuf.Ptr
	args := []unsafe.Pointer{
		unsafe.Pointer(&actPtr), unsafe.Pointer(&weightPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&gridPtr), unsafe.Pointer(&nBlocks), unsafe.Pointer(&outDimU), unsafe.Pointer(&rowBytesU),
	}
	return LaunchKernel(cudaGemvIQ2Opt, uint32(outDim), 1, 1, 256, 1, 1, 256*4, args...)
}
