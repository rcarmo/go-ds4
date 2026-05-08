package gpu

import (
	"fmt"
	"unsafe"
)

// IQ2_XXS GEMV: each block handles one output row (256 threads cooperatively).
// IQ2_XXS block (66 bytes per 256 elements): f16 d (2B) + 8x8B packed data.
// Each 8-byte group: 4 grid indices (1B each) + 4B packed (sign indices + scale).
// Grid table: iq2xxsSignedGrid[256][128][8] int8 — pre-uploaded to GPU.
//
// Activation is f32 (not Q8K — simpler, avoids quantization format on GPU).

const DS4GemvIQ2PTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry iq2xxs_gemv(
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
    @%p1 bra reduce;

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

    mov.u32 %r20, 0;

sub_loop:
    setp.ge.u32 %p3, %r20, 4;
    @%p3 bra sub_done;

    // gridIdx = (aux8_packed >> (r20*8)) & 0xFF
    shl.b32 %r21, %r20, 3;
    shr.u32 %r13, %r10, %r21;
    and.b32 %r13, %r13, 0xFF;

    // signIdx = (aux32_1 >> (r20*7)) & 0x7F
    mul.lo.u32 %r22, %r20, 7;
    shr.u32 %r14, %r11, %r22;
    and.b32 %r14, %r14, 0x7F;

    // grid offset = (gridIdx*128 + signIdx) * 8
    shl.b32 %r15, %r13, 7;
    add.u32 %r15, %r15, %r14;
    shl.b32 %r15, %r15, 3;
    cvt.u64.u32 %rd12, %r15;
    add.u64 %rd13, %rd2, %rd12;

    // act offset for this sub-group: base + sub*8 floats
    shl.b32 %r23, %r20, 5;
    cvt.u64.u32 %rd16, %r23;
    add.u64 %rd17, %rd15, %rd16;

    // 8-element dot: grid[8] x act[8]
    mov.f32 %f4, 0f00000000;
    mov.u32 %r24, 0;
dot8:
    setp.ge.u32 %p3, %r24, 8;
    @%p3 bra dot8_done;
    cvt.u64.u32 %rd18, %r24;
    add.u64 %rd19, %rd13, %rd18;
    ld.global.s8 %si, [%rd19];
    cvt.rn.f32.s32 %f5, %si;
    shl.b32 %r25, %r24, 2;
    cvt.u64.u32 %rd20, %r25;
    add.u64 %rd21, %rd17, %rd20;
    ld.global.f32 %f6, [%rd21];
    fma.rn.f32 %f4, %f5, %f6, %f4;
    add.u32 %r24, %r24, 1;
    bra dot8;
dot8_done:
    fma.rn.f32 %f2, %f3, %f4, %f2;

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

reduce:
    mul.lo.u32 %r10, %r1, 4;
    mov.u64 %rd22, sdata;
    cvt.u64.u32 %rd23, %r10;
    add.u64 %rd24, %rd22, %rd23;
    st.shared.f32 [%rd24], %f0;
    bar.sync 0;
    mov.u32 %r11, 128;
red_loop:
    setp.lt.u32 %p3, %r11, 1;
    @%p3 bra store;
    setp.ge.u32 %p2, %r1, %r11;
    @%p2 bra red_skip;
    add.u32 %r12, %r1, %r11;
    mul.lo.u32 %r13, %r12, 4;
    cvt.u64.u32 %rd23, %r13;
    add.u64 %rd24, %rd22, %rd23;
    ld.shared.f32 %f8, [%rd24];
    mul.lo.u32 %r13, %r1, 4;
    cvt.u64.u32 %rd23, %r13;
    add.u64 %rd24, %rd22, %rd23;
    ld.shared.f32 %f9, [%rd24];
    add.f32 %f9, %f9, %f8;
    st.shared.f32 [%rd24], %f9;
red_skip:
    bar.sync 0;
    shr.u32 %r11, %r11, 1;
    bra red_loop;
store:
    setp.ne.u32 %p2, %r1, 0;
    @%p2 bra done;
    mov.u64 %rd22, sdata;
    ld.shared.f32 %f10, [%rd22];
    ld.param.u64 %rd25, [param_out];
    mul.wide.u32 %rd26, %r0, 4;
    add.u64 %rd27, %rd25, %rd26;
    st.global.f32 [%rd27], %f10;
done:
    ret;
}
`

var cudaGemvIQ2 CUfunction
var cudaGemvIQ2Ready bool
var cudaGridBuf *Buffer

func InitCUDAGemvIQ2(gridData []int8) bool {
	if cudaGemvIQ2Ready {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4GemvIQ2PTX, "iq2xxs_gemv")
	if err != nil {
		fmt.Printf("[gpu] IQ2 GEMV PTX load failed: %v\n", err)
		return false
	}
	cudaGemvIQ2 = fn

	gridSize := len(gridData)
	var gPtr CUdeviceptr
	if err := CuMemAllocRaw(&gPtr, uint64(gridSize)); err != nil {
		return false
	}
	EnsureContext()
	cuMemcpyHtoD(gPtr, unsafe.Pointer(&gridData[0]), uint64(gridSize))
	cudaGridBuf = &Buffer{Ptr: gPtr, Size: gridSize}

	cudaGemvIQ2Ready = true
	fmt.Println("[gpu] DS4 IQ2_XXS GEMV kernel compiled (CUDA PTX)")
	return true
}

func CUDAMatvecIQ2(output, activation *Buffer, weightPtr CUdeviceptr, inDim, outDim, rowBytes int) error {
	if !cudaGemvIQ2Ready {
		return fmt.Errorf("IQ2 kernel not compiled")
	}
	EnsureContext()

	nBlocks := uint32(inDim / 256)
	outDimU := uint32(outDim)
	rowBytesU := uint32(rowBytes)
	actPtr := activation.Ptr
	outPtr := output.Ptr
	gridPtr := cudaGridBuf.Ptr

	args := []unsafe.Pointer{
		unsafe.Pointer(&actPtr),
		unsafe.Pointer(&weightPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&gridPtr),
		unsafe.Pointer(&nBlocks),
		unsafe.Pointer(&outDimU),
		unsafe.Pointer(&rowBytesU),
	}

	return LaunchKernel(cudaGemvIQ2, uint32(outDim), 1, 1, 256, 1, 1, 256*4, args...)
}
