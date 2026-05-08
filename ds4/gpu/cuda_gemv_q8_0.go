package gpu

import (
	"fmt"
	"unsafe"
)

// DS4 Q8_0 GEMV via CUDA PTX — simplified kernel.
// Each block = one output row, 256 threads cooperatively reduce.
// Weight: f16 scale (2B) + int8[32] (34B/block). Activation: f32.

const DS4GemvQ8_0PTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q8_0_f16scale(
    .param .u64 param_act,
    .param .u64 param_wt,
    .param .u64 param_out,
    .param .u32 param_nBlocks,
    .param .u32 param_outDim,
    .param .u32 param_rowBytes
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<12>;
    .reg .f16 %h0;
    .reg .pred %p<4>;
    .reg .s32 %si;
    .reg .u32 %byte;

    .shared .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;    // row
    mov.u32 %r1, %tid.x;      // tid

    ld.param.u32 %r2, [param_outDim];
    setp.ge.u32 %p0, %r0, %r2;
    @%p0 bra done;

    ld.param.u32 %r3, [param_nBlocks];
    ld.param.u32 %r4, [param_rowBytes];
    ld.param.u64 %rd0, [param_act];
    ld.param.u64 %rd1, [param_wt];

    // Row base
    mul.wide.u32 %rd2, %r0, %r4;
    add.u64 %rd3, %rd1, %rd2;

    mov.f32 %f0, 0f00000000;

    // Each thread processes blocks stride-256
    mov.u32 %r5, %r1;
block_loop:
    setp.ge.u32 %p1, %r5, %r3;
    @%p1 bra reduce;

    // Block offset = b * 34
    mul.lo.u32 %r6, %r5, 34;
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd3, %rd4;

    // Load f16 scale
    ld.global.b16 %h0, [%rd5];
    cvt.f32.f16 %f1, %h0;

    // qs base = rd5 + 2, act base = b*32 floats
    add.u64 %rd6, %rd5, 2;
    shl.b32 %r7, %r5, 5;
    mul.wide.u32 %rd7, %r7, 4;
    add.u64 %rd8, %rd0, %rd7;

    // Dot 32 elements: load 1 byte at a time (simple, correct)
    mov.f32 %f2, 0f00000000;
    mov.u32 %r8, 0;
dot_loop:
    setp.ge.u32 %p2, %r8, 32;
    @%p2 bra dot_done;

    // Load int8 from weight
    cvt.u64.u32 %rd9, %r8;
    add.u64 %rd10, %rd6, %rd9;
    ld.global.s8 %si, [%rd10];
    cvt.rn.f32.s32 %f3, %si;

    // Load f32 from activation
    shl.b32 %r9, %r8, 2;
    cvt.u64.u32 %rd11, %r9;
    add.u64 %rd12, %rd8, %rd11;
    ld.global.f32 %f4, [%rd12];

    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r8, %r8, 1;
    bra dot_loop;
dot_done:

    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u32 %r5, %r5, 256;
    bra block_loop;

reduce:
    // Store partial sum to shared memory
    mul.lo.u32 %r10, %r1, 4;
    mov.u64 %rd13, sdata;
    cvt.u64.u32 %rd14, %r10;
    add.u64 %rd15, %rd13, %rd14;
    st.shared.f32 [%rd15], %f0;
    bar.sync 0;

    // Tree reduction
    mov.u32 %r11, 128;
reduce_loop:
    setp.lt.u32 %p3, %r11, 1;
    @%p3 bra store;
    setp.ge.u32 %p2, %r1, %r11;
    @%p2 bra reduce_skip;
    add.u32 %r12, %r1, %r11;
    mul.lo.u32 %r13, %r12, 4;
    cvt.u64.u32 %rd14, %r13;
    add.u64 %rd15, %rd13, %rd14;
    ld.shared.f32 %f5, [%rd15];
    mul.lo.u32 %r13, %r1, 4;
    cvt.u64.u32 %rd14, %r13;
    add.u64 %rd15, %rd13, %rd14;
    ld.shared.f32 %f6, [%rd15];
    add.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd15], %f6;
reduce_skip:
    bar.sync 0;
    shr.u32 %r11, %r11, 1;
    bra reduce_loop;

store:
    setp.ne.u32 %p2, %r1, 0;
    @%p2 bra done;
    mov.u64 %rd13, sdata;
    ld.shared.f32 %f7, [%rd13];
    ld.param.u64 %rd9, [param_out];
    mul.wide.u32 %rd10, %r0, 4;
    add.u64 %rd11, %rd9, %rd10;
    st.global.f32 [%rd11], %f7;

done:
    ret;
}
`

var cudaGemvQ8_0 CUfunction
var cudaGemvQ8_0Ready bool

func InitCUDAGemvQ8_0() bool {
	if cudaGemvQ8_0Ready {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4GemvQ8_0PTX, "gemv_q8_0_f16scale")
	if err != nil {
		fmt.Printf("[gpu] Q8_0 GEMV PTX load failed: %v\n", err)
		return false
	}
	cudaGemvQ8_0 = fn
	cudaGemvQ8_0Ready = true
	fmt.Println("[gpu] DS4 Q8_0 GEMV kernel compiled (CUDA PTX)")
	return true
}

func CUDAMatvecQ8_0(output, activation *Buffer, weightPtr CUdeviceptr, inDim, outDim, rowBytes int) error {
	if !cudaGemvQ8_0Ready {
		return fmt.Errorf("Q8_0 kernel not compiled")
	}
	EnsureContext()

	nBlocks := uint32(inDim / 32)
	outDimU := uint32(outDim)
	rowBytesU := uint32(rowBytes)
	actPtr := activation.Ptr
	outPtr := output.Ptr

	args := []unsafe.Pointer{
		unsafe.Pointer(&actPtr),
		unsafe.Pointer(&weightPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&nBlocks),
		unsafe.Pointer(&outDimU),
		unsafe.Pointer(&rowBytesU),
	}

	return LaunchKernel(cudaGemvQ8_0, uint32(outDim), 1, 1, 256, 1, 1, 256*4, args...)
}

// SwiGLU PTX kernel: dst[i] = silu(a[i]) * b[i] = a[i] * sigmoid(a[i]) * b[i]
const DS4SwiGLUPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry swiglu(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_dst,
    .param .u32 param_n
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r2, %r0, %r1, %r2;
    ld.param.u32 %r3, [param_n];
    setp.ge.u32 %p0, %r2, %r3;
    @%p0 bra done;
    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_dst];
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    // sigmoid(x) = 1/(1+exp(-x))
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // exp(-x) = 2^(-x/ln2) via ex2.approx
    mul.f32 %f2, %f0, 0fBFB8AA3B;  // -1/ln(2) = -1.4426950408
    ex2.approx.f32 %f2, %f2;         // 2^(-x/ln2) = e^(-x)
    add.f32 %f3, %f2, 0f3F800000;    // 1 + exp(-x)
    rcp.approx.f32 %f3, %f3;          // 1/(1+exp(-x)) = sigmoid
    mul.f32 %f4, %f0, %f3;            // x * sigmoid(x) = silu(x)
    mul.f32 %f4, %f4, %f1;            // silu(x) * b
    st.global.f32 [%rd6], %f4;
done:
    ret;
}
`

var cudaSwiGLU CUfunction
var cudaSwiGLUReady bool

func InitCUDASwiGLU() bool {
	if cudaSwiGLUReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4SwiGLUPTX, "swiglu")
	if err != nil {
		fmt.Printf("[gpu] SwiGLU PTX load failed: %v\n", err)
		return false
	}
	cudaSwiGLU = fn
	cudaSwiGLUReady = true
	return true
}

// CUDASwiGLU dispatches dst[i] = silu(a[i]) * b[i] on GPU.
func CUDASwiGLU(dst, a, b *Buffer, n int) error {
	if !cudaSwiGLUReady {
		return fmt.Errorf("SwiGLU not compiled")
	}
	nn := uint32(n)
	aPtr, bPtr, dPtr := a.Ptr, b.Ptr, dst.Ptr
	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr), unsafe.Pointer(&bPtr),
		unsafe.Pointer(&dPtr), unsafe.Pointer(&nn),
	}
	groups := (nn + 255) / 256
	return LaunchKernel(cudaSwiGLU, groups, 1, 1, 256, 1, 1, 0, args...)
}

func CudaSwiGLUReady() bool { return cudaSwiGLUReady }
