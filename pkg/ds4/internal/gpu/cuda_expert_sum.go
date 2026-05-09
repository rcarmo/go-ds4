package gpu

import (
	"fmt"
	"unsafe"
)

const DS4ExpertSumRowsPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry expert_sum_rows_f32(
    .param .u64 param_src,
    .param .u64 param_dst,
    .param .u32 param_width,
    .param .u32 param_rows
) {
    .reg .u32 %r<12>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<4>;
    .reg .pred %p<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2; // column

    ld.param.u32 %r4, [param_width];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_dst];
    ld.param.u32 %r5, [param_rows];

    mov.f32 %f0, 0f00000000;
    mov.u32 %r6, 0;
row_loop:
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra store;
    mul.lo.u32 %r7, %r6, %r4;
    add.u32 %r8, %r7, %r3;
    mul.wide.u32 %rd2, %r8, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.rn.f32 %f0, %f0, %f1;
    add.u32 %r6, %r6, 1;
    bra row_loop;

store:
    mul.wide.u32 %rd4, %r3, 4;
    add.u64 %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f0;

done:
    ret;
}
`

var cudaExpertSumRows CUfunction
var cudaExpertSumRowsReady bool

func InitCUDAExpertSumRows() bool {
	if cudaExpertSumRowsReady {
		return true
	}
	if !gpuOK {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4ExpertSumRowsPTX, "expert_sum_rows_f32")
	if err != nil {
		fmt.Printf("[gpu] expert row-sum PTX load failed: %v\n", err)
		return false
	}
	cudaExpertSumRows = fn
	cudaExpertSumRowsReady = true
	fmt.Println("[gpu] DS4 expert f32 row-sum kernel compiled")
	return true
}

func CUDAExpertSumRows(src, dst *Buffer, width, rows int, stream CUstream) error {
	if !cudaExpertSumRowsReady {
		return fmt.Errorf("expert sum rows kernel not compiled")
	}
	if rows <= 0 || width <= 0 {
		return nil
	}
	srcPtr := src.Ptr
	dstPtr := dst.Ptr
	widthU := uint32(width)
	rowsU := uint32(rows)
	args := []unsafe.Pointer{
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&widthU),
		unsafe.Pointer(&rowsU),
	}
	grid := uint32((width + 255) / 256)
	if stream != 0 {
		return LaunchKernelStream(cudaExpertSumRows, grid, 1, 1, 256, 1, 1, 0, stream, args...)
	}
	return LaunchKernel(cudaExpertSumRows, grid, 1, 1, 256, 1, 1, 0, args...)
}
