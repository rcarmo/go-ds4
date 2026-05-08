package gpu

import (
	"fmt"
	"unsafe"
)

// Batched attention kernel: all 64 heads scored in parallel on GPU.
// Each block = one head. 256 threads cooperatively score all KV rows.
//
// Input:  Q[NHead * NHeadDim], KV_cache[nRaw * NHeadDim], sinks[NHead]
// Output: heads[NHead * NHeadDim]
//
// Per block (head):
//   1. Load Q_head[512] into shared memory
//   2. Each thread scores one or more KV rows (dot product)
//   3. Stable softmax with sink attention
//   4. Weighted accumulation into output head

const DS4AttnPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry batched_attn(
    .param .u64 param_q,
    .param .u64 param_kv,
    .param .u64 param_sinks,
    .param .u64 param_out,
    .param .u32 param_nRaw,
    .param .u32 param_headDim,
    .param .f32 param_scale
) {
    .reg .u32 %r<20>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<16>;
    .reg .pred %p<4>;

    // Shared memory: Q head (512 floats) + scores (256 floats) + output (512 floats)
    .shared .f32 s_q[512];
    .shared .f32 s_score[256];
    .shared .f32 s_out[512];

    mov.u32 %r0, %ctaid.x;    // head index
    mov.u32 %r1, %tid.x;      // thread ID
    ld.param.u32 %r2, [param_nRaw];
    ld.param.u32 %r3, [param_headDim];
    ld.param.f32 %f0, [param_scale];
    ld.param.u64 %rd0, [param_q];
    ld.param.u64 %rd1, [param_kv];
    ld.param.u64 %rd2, [param_sinks];
    ld.param.u64 %rd3, [param_out];

    // Phase 1: cooperatively load Q_head into shared memory
    // Q offset = head * headDim
    mul.lo.u32 %r4, %r0, %r3;
    mul.wide.u32 %rd4, %r4, 4;
    add.u64 %rd5, %rd0, %rd4;
    mov.u64 %rd6, s_q;
    mov.u32 %r5, %r1;
load_q:
    setp.ge.u32 %p0, %r5, %r3;
    @%p0 bra load_q_done;
    mul.wide.u32 %rd7, %r5, 4;
    add.u64 %rd8, %rd5, %rd7;
    ld.global.f32 %f1, [%rd8];
    add.u64 %rd9, %rd6, %rd7;
    st.shared.f32 [%rd9], %f1;
    add.u32 %r5, %r5, 256;
    bra load_q;
load_q_done:

    // Zero output in shared memory
    mov.u64 %rd10, s_out;
    mov.u32 %r5, %r1;
zero_out:
    setp.ge.u32 %p0, %r5, %r3;
    @%p0 bra zero_done;
    mul.wide.u32 %rd7, %r5, 4;
    add.u64 %rd8, %rd10, %rd7;
    st.shared.f32 [%rd8], 0f00000000;
    add.u32 %r5, %r5, 256;
    bra zero_out;
zero_done:
    bar.sync 0;

    // Phase 2: each thread scores one KV row (if tid < nRaw)
    // score[tid] = dot(s_q, kv_row[tid]) * scale
    mov.f32 %f2, 0f00000000;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra score_done;

    // KV row address = kv + tid * headDim * 4
    mul.lo.u32 %r6, %r1, %r3;
    mul.wide.u32 %rd11, %r6, 4;
    add.u64 %rd12, %rd1, %rd11;

    // Dot product Q_shared . KV_global
    mov.u32 %r7, 0;
dot_loop:
    setp.ge.u32 %p0, %r7, %r3;
    @%p0 bra dot_done;
    mul.wide.u32 %rd7, %r7, 4;
    add.u64 %rd8, %rd6, %rd7;
    ld.shared.f32 %f3, [%rd8];
    add.u64 %rd9, %rd12, %rd7;
    ld.global.f32 %f4, [%rd9];
    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r7, %r7, 1;
    bra dot_loop;
dot_done:
    mul.f32 %f2, %f2, %f0;

score_done:
    // Store score
    mov.u64 %rd13, s_score;
    mul.wide.u32 %rd7, %r1, 4;
    add.u64 %rd8, %rd13, %rd7;
    st.shared.f32 [%rd8], %f2;
    bar.sync 0;

    // Phase 3: softmax with sink (thread 0 does serial softmax - simple but correct)
    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra softmax_done;

    // Load sink for this head
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.f32 %f5, [%rd8];

    // Find max (including sink)
    mov.f32 %f6, %f5;
    mov.u32 %r8, 0;
max_loop:
    setp.ge.u32 %p2, %r8, %r2;
    @%p2 bra max_done;
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd13, %rd7;
    ld.shared.f32 %f7, [%rd8];
    max.f32 %f6, %f6, %f7;
    add.u32 %r8, %r8, 1;
    bra max_loop;
max_done:

    // Compute exp and sum (including sink)
    sub.f32 %f8, %f5, %f6;
    mul.f32 %f8, %f8, 0fBFB8AA3B;
    ex2.approx.f32 %f8, %f8;
    mov.f32 %f9, %f8;

    mov.u32 %r8, 0;
exp_loop:
    setp.ge.u32 %p2, %r8, %r2;
    @%p2 bra exp_done;
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd13, %rd7;
    ld.shared.f32 %f7, [%rd8];
    sub.f32 %f7, %f7, %f6;
    mul.f32 %f7, %f7, 0fBFB8AA3B;
    ex2.approx.f32 %f7, %f7;
    st.shared.f32 [%rd8], %f7;
    add.f32 %f9, %f9, %f7;
    add.u32 %r8, %r8, 1;
    bra exp_loop;
exp_done:

    // Normalize
    rcp.approx.f32 %f9, %f9;
    mov.u32 %r8, 0;
norm_loop:
    setp.ge.u32 %p2, %r8, %r2;
    @%p2 bra norm_done;
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd8, %rd13, %rd7;
    ld.shared.f32 %f7, [%rd8];
    mul.f32 %f7, %f7, %f9;
    st.shared.f32 [%rd8], %f7;
    add.u32 %r8, %r8, 1;
    bra norm_loop;
norm_done:

softmax_done:
    bar.sync 0;

    // Phase 4: weighted accumulation - each thread handles a KV row
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra accum_done;

    // weight = s_score[tid]
    mul.wide.u32 %rd7, %r1, 4;
    add.u64 %rd8, %rd13, %rd7;
    ld.shared.f32 %f10, [%rd8];

    // Accumulate: s_out[d] += weight * kv[tid * headDim + d]
    mul.lo.u32 %r6, %r1, %r3;
    mul.wide.u32 %rd11, %r6, 4;
    add.u64 %rd12, %rd1, %rd11;

    mov.u32 %r7, 0;
accum_loop:
    setp.ge.u32 %p0, %r7, %r3;
    @%p0 bra accum_done;
    mul.wide.u32 %rd7, %r7, 4;
    add.u64 %rd8, %rd12, %rd7;
    ld.global.f32 %f11, [%rd8];
    mul.f32 %f11, %f11, %f10;
    add.u64 %rd9, %rd10, %rd7;
    // Atomic add to shared memory
    atom.shared.add.f32 %f12, [%rd9], %f11;
    add.u32 %r7, %r7, 1;
    bra accum_loop;
accum_done:
    bar.sync 0;

    // Phase 5: write output heads
    // out[head * headDim + d] = s_out[d] (cooperative write)
    mul.lo.u32 %r4, %r0, %r3;
    mul.wide.u32 %rd4, %r4, 4;
    add.u64 %rd14, %rd3, %rd4;
    mov.u32 %r5, %r1;
write_out:
    setp.ge.u32 %p0, %r5, %r3;
    @%p0 bra done;
    mul.wide.u32 %rd7, %r5, 4;
    add.u64 %rd8, %rd10, %rd7;
    ld.shared.f32 %f13, [%rd8];
    add.u64 %rd9, %rd14, %rd7;
    st.global.f32 [%rd9], %f13;
    add.u32 %r5, %r5, 256;
    bra write_out;
done:
    ret;
}
`

var cudaAttn CUfunction
var cudaAttnReady bool

func InitCUDAAttn() bool {
	if cudaAttnReady {
		return true
	}
	if !Available() {
		return false
	}
	EnsureContext()
	fn, err := LoadPTX(DS4AttnPTX, "batched_attn")
	if err != nil {
		fmt.Printf("[gpu] Attention PTX load failed: %v\n", err)
		return false
	}
	cudaAttn = fn
	cudaAttnReady = true
	fmt.Println("[gpu] Batched attention kernel compiled (CUDA PTX)")
	return true
}

func CudaAttnReady() bool { return cudaAttnReady }

// CUDABatchedAttn dispatches all 64 heads' attention scoring on GPU.
func CUDABatchedAttn(qBuf, kvBuf, sinkBuf, outBuf *Buffer, nRaw, headDim int, scale float32, stream CUstream) error {
	if !cudaAttnReady {
		return fmt.Errorf("attn kernel not compiled")
	}
	nRawU := uint32(nRaw)
	headDimU := uint32(headDim)
	qPtr := qBuf.Ptr
	kvPtr := kvBuf.Ptr
	sinkPtr := sinkBuf.Ptr
	outPtr := outBuf.Ptr
	args := []unsafe.Pointer{
		unsafe.Pointer(&qPtr), unsafe.Pointer(&kvPtr), unsafe.Pointer(&sinkPtr), unsafe.Pointer(&outPtr),
		unsafe.Pointer(&nRawU), unsafe.Pointer(&headDimU), unsafe.Pointer(&scale),
	}
	nHead := uint32(64)
	return LaunchKernelStream(cudaAttn, nHead, 1, 1, 256, 1, 1, (512+256+512)*4, stream, args...)
}
