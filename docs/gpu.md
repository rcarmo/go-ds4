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

### Always on GPU (when available)
- **Q8_0 projections** with outDim ≥ 4096:
  - `output.weight` (4096 → 129280) — the single largest matmul
  - `attn_q_b.weight` (1024 → 32768) — per layer
  - `attn_output_b.weight` (1024 → 4096) — per layer
  - `ffn_down_shexp.weight` (2048 → 4096) — per layer

### GPU with expert cache (demand-filled)
- **IQ2_XXS gate/up** projections for top-4 routed experts
- **Q2_K down** projections for top-4 routed experts
- **SwiGLU** activation (fused on GPU, no CPU round-trip)

### Stays on CPU
- Small Q8_0 projections (outDim < 4096): attn_q_a, attn_kv
- Attention scoring (Sdot + Saxpy — too small for GPU dispatch)
- RMSNorm, RoPE, softmax
- KV cache operations
- Compressor / indexer projections
- HC pre/post mixing

## VRAM Budget

| Component | Size | Notes |
|---|---|---|
| Q8_0 dense projections | 3.2 GB | 130 tensors, uploaded at init |
| Expert cache (demand) | 0–4.5 GB | Fills as experts are routed |
| Batched expert buffers | 27 MB | Gate+up+down for 4 experts |
| IQ2 grid table | 256 KB | Uploaded once at init |
| Transient buffers | ~10 MB | Activation, output, intermediate |
| **Total** | **3.5–8 GB** | Fits in 12 GB GPU |

## Performance Characteristics

| Metric | Value |
|---|---|
| GPU kernel time (IQ2 4096→2048) | 2.4 µs |
| GPU kernel time (Q2K 2048→4096) | 3 µs |
| GPU kernel time (Q8_0 4096→4096) | 3 µs |
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
