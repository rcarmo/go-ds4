# go-ds4 Architecture

## Overview

go-ds4 is a pure Go inference engine for DeepSeek V4 Flash, a 128GB MoE language model with 43 transformer layers, 256 routed experts, and a novel MLA (Multi-head Latent Attention) mechanism. It runs on CPU (AVX2/NEON SIMD) and optionally NVIDIA GPU (CUDA PTX), all without CGo.

## Inference Flow

```
Input text
  → GPT-2 BPE tokenizer (JoyAI pre-tokenizer)
  → Token IDs
  → F16 embedding lookup (mmap'd)
  → 43 transformer layers:
      ┌─ HC pre (Sinkhorn 4-stream mixing)
      ├─ MLA Attention
      │   ├─ Q LoRA: 4096 → 1024 → 32768 (Q8_0)
      │   ├─ KV projection: 4096 → 512 (Q8_0)
      │   ├─ Per-head RMSNorm + RoPE (YaRN tail-only)
      │   ├─ FP8 KV quantization → ring cache push
      │   ├─ Compressor update (ratio-4/128 layers)
      │   ├─ Indexer top-K gating (ratio-4 layers)
      │   ├─ Mixed attention: SWA raw + compressed + sink
      │   └─ Output LoRA: grouped 8-way → 1024 → 4096 (Q8_0)
      ├─ HC post (residual injection)
      ├─ HC pre (Sinkhorn, FFN sublayer)
      ├─ MoE FFN
      │   ├─ Expert routing: softmax → top-4/6 selection
      │   ├─ Routed experts (GPU or CPU):
      │   │   ├─ Gate: IQ2_XXS [2048, 4096]
      │   │   ├─ Up:   IQ2_XXS [2048, 4096]
      │   │   ├─ SwiGLU activation
      │   │   └─ Down:  Q2_K [4096, 2048]
      │   └─ Shared expert (Q8_0, runs concurrently)
      └─ HC post (routed + shared → 4-stream state)
  → HC collapse (sigmoid-weighted 4-stream sum)
  → RMSNorm
  → Output projection: Q8_0 [4096, 129280]
  → Logits → Sampling (temperature/top-k/top-p/min-p)
  → Output token
```

## Quantization Formats

| Format | Block Size | Bytes/Block | Elements/Block | Use |
|---|---|---|---|---|
| **Q8_0** (DS4) | 34 | 2 (F16 scale) + 32 (int8) | 32 | Dense projections, shared expert |
| **Q2_K** | 84 | 16 (scales) + 64 (packed 2-bit) + 4 (F16 d/dmin) | 256 | Expert down projection |
| **IQ2_XXS** | 66 | 2 (F16 d) + 64 (grid indices) | 256 | Expert gate/up projections |
| **Q8_K** | 292 | 4 (F32 d) + 256 (int8) + 32 (bsums) | 256 | Activation quantization |
| **F16** | 2 | — | 1 | Token embeddings, HC mixing |
| **FP8 E4M3** | 1 | — | 1 | KV cache (non-RoPE portion) |

Note: DS4's Q8_0 uses **F16 scales** (34-byte blocks), not the standard ggml Q8_0 with F32 scales (36-byte blocks). This affects all SIMD and GPU kernels.

## GPU Architecture

### Backend Selection
```
if NVIDIA GPU + libcuda.so.1 available:
    → CUDA PTX (6 hand-written kernels)
elif Vulkan + libvulkan.so.1 available:
    → Vulkan SPIR-V (2 compiled shaders)
else:
    → CPU SIMD (AVX2 or NEON assembly)
```

All GPU backends load via `purego.Dlopen` at runtime — no CGo, no build-time GPU dependencies.

### CUDA PTX Kernels

| Kernel | PTX File | Description |
|---|---|---|
| `gemv_q8_0_f16scale` | `cuda_gemv_q8_0.go` | Q8_0 GEMV: F16 scale decode + int8×f32 dot, 256-thread shared-memory reduction |
| `iq2xxs_gemv_opt` | `cuda_gemv_iq2_opt.go` | IQ2_XXS GEMV: **shared memory activation tiling** (cooperative 16KB load), vectorized grid loads (`ld.global.v2.u32`), warp shuffle reduction. 2.4µs/call |
| `q2k_gemv` | `cuda_gemv_q2k.go` | Q2_K GEMV: 2-bit extraction from packed bytes, per-group scale application |
| `swiglu` | `cuda_gemv_q8_0.go` | Fused SiLU×mul: `ex2.approx` + `rcp.approx` for fast sigmoid |

### Shared Memory Activation Tiling

The IQ2 kernel uses cooperative shared memory loading:

1. **Phase 1**: 256 threads cooperatively load `activation[4096]` into `s_act[4096]` (16KB shared memory). Each thread loads 16 elements (stride 256). One `bar.sync` barrier.
2. **Phase 2**: All dot product computations read activation from `ld.shared.f32` (1 cycle latency) instead of `ld.global.f32` (~300 cycles). Each activation element is read by potentially all 256 threads — shared memory eliminates 256× redundant global reads.

Result: 3.0µs → 2.4µs per kernel call (20% faster).

### Batched Expert Pipeline

Per layer, the 4 active experts are dispatched as one batch:

1. **DtoD copy** (~70µs): Concatenate 4 experts' cached weights contiguously
2. **IQ2 gate** (1 launch, 8192 rows): All 4 experts' gate projections
3. **IQ2 up** (1 launch, 8192 rows): All 4 experts' up projections
4. **SwiGLU** (1 launch, 8192 elements): Fused activation on GPU
5. **Q2K down** (1 launch, 16384 rows): All 4 experts' down projections
6. **Sync + Download** (1 per layer): Single `cuCtxSynchronize`

Total: 4 kernel launches + 1 sync per layer × 43 layers = 172 launches + 43 syncs per token.

### CUDA Streams

All expert GPU operations run on a dedicated CUDA stream:
- `cuMemcpyHtoDAsync` for activation upload
- `cuLaunchKernel` with stream parameter for all 4 kernels
- `cuStreamSynchronize` instead of `cuCtxSynchronize` (only waits for our work)
- `cuMemcpyDtoHAsync` for result download

This avoids blocking other GPU work and reduces sync overhead.

### Expert VRAM Cache

- **Demand-filled**: experts are uploaded to VRAM on first route, not pre-loaded
- **16 slots per layer**: fits ~4.5 GB for the most-used experts
- **Zero PCIe after warmup**: cached experts dispatch without host↔device transfer
- **Miss fallback**: uncached experts use CPU AVX2 path

## KV Cache

### Ring Buffer Design

All three KV cache levels use logical ring buffers (no memmove):

| Level | Capacity | Ring Cursor | Content |
|---|---|---|---|
| **Raw SWA** | 128 rows | `RawWrite` | FP8 quantized KV (non-RoPE) + F32 RoPE tail |
| **Compressed** | ctx/ratio + 2 | `CompWrite` | Learned compressor output, ratio-4 or ratio-128 |
| **Indexer** | ctx/4 + 2 | `IndexCompWrite` | Auxiliary indexer for ratio-4 layers |

### Compression Schedule (from ds4.c)

| Layers | Ratio | Compressor | Indexer |
|---|---|---|---|
| 0–1 | None | — | — |
| Even ≥ 2 | 4:1 | ✓ (dual-lane Sinkhorn pool) | ✓ (top-K selection) |
| Odd ≥ 3 | 128:1 | ✓ (single-lane Sinkhorn pool) | — |

## Hyper-Connections

DeepSeek V4 uses 4 parallel residual streams instead of a single residual. Before each sublayer (attention or FFN), the HC pre-step:

1. Flatten 4 streams → RMSNorm
2. Project through F16 weight → 24 mixing parameters
3. Sinkhorn split: `pre[4]` (input weights), `post[4]` (output weights), `comb[4×4]` (cross-stream mix)
4. 20 iterations of row/column normalization on the combine matrix
5. Weighted sum of streams → sublayer input

After the sublayer, HC post injects the output back into all 4 streams using `post` and `comb` weights.

## Session Persistence

Binary payload format (v4):

```
Header: magic(4) + version(4) + pos(4) + ctxSize(4)
Per layer (×43):
  nRaw(4) + rawWrite(4) + rawKV[capRaw × NHeadDim × 4]
  if compressed:
    nComp(4) + compWrite(4) + compKV[...] + compStateKV[...] + compStateScore[...]
    if ratio-4:
      nIndexComp(4) + indexCompWrite(4) + indexCompKV[...] + indexStateKV[...] + indexStateScore[...]
Tokens: nTokens(4) + tokenIDs[nTokens × 4]
```

## Memory Modes

| Mode | `EngineOptions` | RSS | VRAM | Expert Access |
|---|---|---|---|---|
| Default | `{}` | ~7 GB | 0 | mmap demand-page |
| GPU | `{UseGPU: true}` | ~7 GB | ~8 GB | Demand-cache in VRAM |
| Pin | `{PinNonExpert: true}` | ~6.5 GB | 0 | mmap, dense mlocked |
| Evict | `{EvictExperts: true}` | varies | 0 | MADV_DONTNEED cold pages |
| Stream | `{StreamExperts: true}` | ~2 GB | 0 | pread from NVMe |
| Fast | `{FastExperts: true}` | same | same | Top-4 instead of top-6 |
