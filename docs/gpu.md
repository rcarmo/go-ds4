# go-ds4 GPU Guide

## Requirements

- **NVIDIA GPU** with CUDA driver (any version, compute capability ≥ 8.0 for PTX sm_80)
- **libcuda.so.1** available at runtime (installed with NVIDIA driver, no CUDA toolkit needed)
- No CGo for CUDA/Vulkan — GPU bindings use `purego.Dlopen` to load the CUDA driver API at runtime

For Vulkan (fallback, portable):
- **libvulkan.so.1** available at runtime
- Any Vulkan-capable GPU (NVIDIA, AMD, Intel, ARM Mali)

For macOS Metal:
- Build with `CGO_ENABLED=1 go build -tags metal`.
- Select with `DS4_GPU_BACKEND=metal`.
- See [Metal experiments](metal.md) for the constrained-memory path, current measurements, and limitations.

## Usage

```go
engine, _ := ds4.OpenEngineWithOptions(ds4.EngineOptions{
    ModelPath:   "gguf/ds4-q2.gguf",
    UseGPU:      true,   // enable GPU acceleration
    FastExperts: true,    // top-4 experts (recommended with GPU)
})
```

Or via the server:
```bash
./ds4server -model gguf/ds4-q2.gguf -listen :8080 -fast
```

## What Gets Accelerated

### Default correctness mode

Normal `UseGPU: true` now enables the parity-safe CUDA Q8_0 dense path:

- Activations are quantized on CPU with the same C-style Q8_0 prequantization used by the coherent CPU path.
- CUDA consumes `xq int8[] + xscale float32[]` and computes `Σ f16(wd) * xscale * dot_i8(wq, xq)`, matching `dotQ8_0Prequant`.
- Q8_0 projections with outDim ≥ 2048 are uploaded and dispatched on GPU, including `output.weight`, `attn_q_b`, grouped `attn_output_b` (8192 → 4096), and shared expert projections.

The old routed expert/fused/Vulkan kernels are still disabled by default because they do not yet implement the same expert math:

- Expert kernels consume F32 activations and apply router weights after the Q2_K down path, while the parity path uses Q8_K activation quantization, clamps/weights before Q8_K hidden quantization, and the exact ds4.c Q2_K shift traversal.
- Fused Q8_0 `attn_q_a + attn_kv` inherits the old F32-activation Q8_0 mismatch.

### Unsafe legacy GPU mode

Set `DS4_UNSAFE_GPU_NONPARITY=1` to re-enable the old experimental kernels for benchmarking only. In that mode:

- Legacy fused Q8_0 `attn_q_a + attn_kv` may run on GPU.
- Legacy **IQ2_XXS/Q2_K routed expert cache** may run on GPU.
- Legacy **SwiGLU** may run on GPU.

### Still CPU in correctness mode

- IQ2_XXS/Q2_K routed experts
- RMSNorm, RoPE, softmax
- KV cache operations
- Compressor / indexer projections
- HC pre/post mixing

## VRAM Budget

| Component | Size | Notes |
|---|---|---|
| Q8_0 dense projections | ~6.0 GB | Default CUDA parity path |
| Expert cache (demand) | 0–4.5 GB | Unsafe legacy mode only |
| Batched expert buffers | 27 MB | Unsafe legacy mode only |
| IQ2 grid table | 256 KB | Unsafe legacy mode only |
| Transient buffers | ~10 MB | Activation, output, intermediate |
| **Total** | **~6 GB default / 6–10 GB unsafe** | Fits in 12 GB GPU |

## Strict CUDA Compact Expert Cache

Strict CUDA uses a compact routed-expert cache inspired by the `feature/mlx-flash` branch:

- selected `gate`, `up`, and `down` expert slices are cached individually in VRAM with LRU eviction;
- per-token top-k expert batches are assembled with device-to-device copies;
- host-to-device expert uploads happen only on cache misses;
- close-time stats report live/peak cache bytes, entries, hits, misses, evictions, H→D bytes, and D→D bytes.

Control the cache with:

```bash
DS4_CUDA_COMPACT_EXPERT_CACHE_MB=0      # default: disabled
DS4_CUDA_COMPACT_EXPERT_CACHE_MB=2048   # opt-in 2 GiB cache for A/B tests
```

The cache is currently opt-in. On a 12 GB RTX 3060-class GPU strict mode already keeps ~6.9 GB of Q8_0 dense weights resident, and a 2 GiB compact expert cache thrashes heavily for short prompts. On cache misses it uploads to resident cache and then D→D copies into the batch buffer, so low hit-rate runs can be slower than direct H→D assembly.

Post-upstream-merge 8-token strict greedy smoke on RTX 3060, prompt `Hi`, two repeated runs:

| Mode | Prefill | Decode | Total | Cache stats |
|---|---:|---:|---:|---|
| cache disabled | 0.6 tok/s | 0.9–1.0 tok/s | 16.7–17.8s | direct H→D for every selected expert |
| 2048 MB compact cache | 0.8–0.9 tok/s | 0.5–0.7 tok/s | 17.9–22.7s | ~3.8–3.9k hits, ~6.2k misses, ~5.3k evictions, ~14 GB H→D, 22.6 GB D→D |

Conclusion: the previous single-run 2048 MB cache win was not stable. The implementation remains useful for experiments and larger-cache systems, but it is not the default performance path on 12 GB CUDA until hit rate improves or miss handling avoids double-copy cost.

## Performance Characteristics

| Metric | Value |
|---|---|
| Legacy GPU kernel time (IQ2 4096→2048) | 2.4 µs |
| Legacy GPU kernel time (Q2K 2048→4096) | 3 µs |
| Legacy GPU kernel time (Q8_0 4096→4096) | 3 µs |
| DtoD expert weight copy (4 experts) | ~70 µs |
| Host→Device activation upload | 8 µs |
| Device→Host result download | 10 µs |
| cuCtxSynchronize overhead | ~25 µs |
| **Total per layer (batched, 4 experts)** | **~100 µs** |
| **Total per token (43 layers)** | **~4.3 ms** |

## Correctness Contract

`-gpu-strict` is the correctness harness for V4 GPU work. It means:

- No silent CPU fallback for GPU-covered V4 paths.
- Deterministic strict-GPU execution.
- Kernel-level parity tests for Q8_0 prequant, Q2_K×Q8_K, IQ2_XXS×Q8_K, and fused expert SwiGLU→Q8_K→down.
- Full-model CPU-vs-strict-GPU equivalence is bounded numerical equivalence, not bit identity:
  - greedy argmax must match,
  - top-10 logits must overlap by at least 8/10,
  - RMSE must remain below 0.35,
  - max logit drift must remain below 2.0 for the checked prefix.

Run the full model check explicitly:

```bash
DS4_RUN_MODEL_PARITY=1 TMPDIR=/workspace/tmp go test ./pkg/ds4 -run TestV4StrictGPUCPUEquivalence -v
```

Set `DS4_MODEL=/path/to/model.gguf` to override the default `gguf/ds4-q2.gguf`.

Bit-for-bit CPU/GPU identity is not the production goal: it requires serial CPU-order kernels and disables the parallel reductions that make GPU useful. Strict mode instead locks down deterministic bounded equivalence plus direct quant-kernel parity.

## Limitations

1. **Single-token decode**: GPU dispatch overhead (~100µs/layer) limits speedup at single-token granularity. GPU wins more on prefill (multiple tokens).

2. **PCIe 3.0 bandwidth**: Expert weights demand-cached at ~12 GB/s PCIe. First route to a new expert costs ~500µs upload. Subsequent routes are instant.

3. **Vulkan path**: The Vulkan SPIR-V shaders compile but the `llvmpipe` software renderer crashes on some SPIR-V; only works with real GPU Vulkan drivers (NVIDIA proprietary, Mesa radeonsi).

4. **No multi-GPU**: Single GPU only. The full expert set (72 GB) doesn't fit in 12 GB VRAM; we cache the most-used ~16 per layer.

## PTX Kernels

All kernels target `sm_80` (Ampere) and use:
- **Shared memory activation tiling**: cooperative load of activation vector into 16KB shared memory (IQ2 kernel), 256× fewer global reads
- `ld.global.v2.u32` for vectorized grid/weight loads
- `ld.global.b16` + `cvt.f32.f16` for F16 scale decode
- Warp shuffle reduction (`shfl.sync.down`) instead of shared memory tree
- `ex2.approx` + `rcp.approx` for fast SiLU sigmoid
- CUDA stream dispatch for all kernels + async memcpy

Source: all PTX is embedded as Go string constants in `pkg/ds4/internal/gpu/cuda_gemv_*.go`. No external `.ptx` files or nvcc dependency.
