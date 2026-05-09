# go-ds4 GPU Guide

## Requirements

- **NVIDIA GPU** with CUDA driver (any version, compute capability ≥ 8.0 for PTX sm_80)
- **libcuda.so.1** available at runtime (installed with NVIDIA driver, no CUDA toolkit needed)
- No CGo — GPU bindings use `purego.Dlopen` to load the CUDA driver API at runtime

For Vulkan (fallback, portable):
- **libvulkan.so.1** available at runtime
- Any Vulkan-capable GPU (NVIDIA, AMD, Intel, ARM Mali)

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

After the CPU/C parity fixes, the old CUDA/Vulkan Q8_0 and routed expert kernels are **disabled by default** because they do not implement the same math:

- Q8_0 dense kernels consume F32 activations, while the parity path uses C-style Q8_0 activation prequantization.
- Expert kernels consume F32 activations and apply router weights after the Q2_K down path, while the parity path uses Q8_K activation quantization, clamps/weights before Q8_K hidden quantization, and the exact ds4.c Q2_K shift traversal.
- Fused Q8_0 `attn_q_a + attn_kv` inherits the same Q8_0 mismatch.

With normal `UseGPU: true`, GPU initialization is refused and the engine falls back to CPU so generation remains coherent.

### Unsafe legacy GPU mode

Set `DS4_UNSAFE_GPU_NONPARITY=1` to re-enable the old experimental kernels for benchmarking only. In that mode:

- **Q8_0 projections** with outDim ≥ 4096 may run on GPU:
  - `output.weight` (4096 → 129280)
  - `attn_q_b.weight` (1024 → 32768)
  - `attn_output_b.weight` (8192 → 4096)
  - `ffn_down_shexp.weight` (2048 → 4096)
- **IQ2_XXS/Q2_K routed expert cache** may run on GPU.
- **SwiGLU** may run on GPU.

### Still CPU in correctness mode

- Q8_0 dense projections
- IQ2_XXS/Q2_K routed experts
- RMSNorm, RoPE, softmax
- KV cache operations
- Compressor / indexer projections
- HC pre/post mixing

## VRAM Budget

| Component | Size | Notes |
|---|---|---|
| CUDA/Vulkan buffers | 0 MB | Default correctness mode refuses GPU init |
| Q8_0 dense projections | 3.2 GB | Unsafe legacy mode only |
| Expert cache (demand) | 0–4.5 GB | Unsafe legacy mode only |
| Batched expert buffers | 27 MB | Unsafe legacy mode only |
| IQ2 grid table | 256 KB | Unsafe legacy mode only |
| Transient buffers | ~10 MB | Activation, output, intermediate |
| **Total** | **0 MB default / 3.5–8 GB unsafe** | Fits in 12 GB GPU |

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

Source: all PTX is embedded as Go string constants in `ds4/gpu/cuda_gemv_*.go`. No external `.ptx` files or nvcc dependency.
