# go-ds4 Performance Optimization History

## Timeline

### Phase 1: Scalar Go (baseline)
- **9.9 s/token** — pure Go scalar dot products, sequential experts
- All quantized math in Go loops

### Phase 2: AVX2 SIMD kernels
- DotQ8_0F32: VPMOVSXBD + VCVTDQ2PS + FMA → **572 ns/4096-dim**
- DotF16F32: VCVTPH2PS + FMA → **528 ns/16384-dim**
- DotI8: VPMADDUBSW + VPMADDWD → **6 ns/32-elem**
- **7.9 s/token** (scalar → SIMD = 1.25×)

### Phase 3: Parallel experts
- 6 experts run concurrently via goroutines with per-expert scratch buffers
- **1.7 s/token** (4.6× speedup from parallelism)

### Phase 4: Fast math + Q2K/IQ2 SIMD
- Schraudolph fast exp for SiLU/sigmoid
- Q2K: DotQ2Group16 AVX2 (VPAND + VPSRLW + VPMADDUBSW)
- IQ2: DotIQ2Group32 AVX2 (zero-copy grid lookup, VINSERTI128)
- **0.86 s/token** = **11.5× total speedup**

### Phase 5: Forward pass correctness
- Fixed GGUF alignment (32-byte default, not 64)
- Fixed Q8_0 block size (34 bytes with F16 scale, not 36 with F32)
- Fixed HC Sinkhorn split (20-iteration row/col normalization)
- Fixed HC pre wiring (AttnCur/FfnCur separation)
- Model now produces meaningful logits

### Phase 6: Ring KV cache + allocation cleanup
- Logical ring buffers for raw + compressed + indexer KV caches
- Eliminated per-token allocations in hot path
- Session persistence v4 with ring cursors
- **~1.1 s/token** (decode, warm)

### Phase 7: GPU acceleration
- CUDA PTX Q8_0 GEMV: **5.5× faster than CPU** for large matmuls
- Vulkan SPIR-V shaders (portable, pending driver compat)
- Selective upload: only outDim ≥ 4096 tensors → 3.2 GB VRAM
- **~1.57 tok/s** with Q8_0-only GPU

### Phase 8: Full GPU expert pipeline
- CUDA PTX IQ2_XXS + Q2_K + SwiGLU kernels
- Demand-filled expert VRAM cache (zero PCIe after warmup)
- Batched 4-expert dispatch (DtoD concat + single kernel launch)
- 43 syncs/token (down from 344)
- **~2.13 tok/s** (1.6× over CPU)

## Key Decisions

| Decision | Rationale |
|---|---|
| F16 scale Q8_0 (34B blocks) | Matches DS4 GGUF format exactly (not ggml's 36B) |
| VCVTPH2PS for F16 decode | Available on all AVX2 CPUs (F16C extension) |
| Unsigned offset trick for int8×int8 | VPMADDUBSW requires unsigned×signed; +128 correction |
| Top-4 expert mode | 33% less expert compute, ~17% faster, slight quality tradeoff |
| Demand-fill expert cache | Static top-N had 0% hit rate — routing doesn't follow index order |
| Batched expert DtoD concat | 4 kernel launches/layer instead of 12; one sync instead of 4 |
| GPU SwiGLU kernel | Eliminates CPU round-trip between gate/up and down projections |
| Logical ring KV cache | O(1) push vs O(N) memmove for sliding window |

## What Didn't Work

| Attempt | Result | Why |
|---|---|---|
| Prequant I8×I8 for Q8_0 | Slower | Per-block scalar f16×f32 chain serializes the loop |
| Vulkan on this container | Crashes | NVIDIA ICD broken; llvmpipe segfaults on SPIR-V |
| Static expert cache (top-16 by index) | 0% hit rate | Routing selects experts 69, 209, 235, etc. |
| Per-expert GPU dispatch | 1.31 tok/s | 344 syncs/token overwhelms kernel speedup |
| Full VRAM expert upload (6.9 GB) | No faster | Most uploaded tensors had outDim < threshold |
