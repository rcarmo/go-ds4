# Implementation Notes

This document collects implementation details that should not crowd the top-level README. For deeper subsystem notes, see:

- [Architecture](architecture.md)
- [GPU backends](gpu.md)
- [Metal experiments](metal.md)
- [Optimization history](optimization.md)
- [SIMD kernels](simd.md)

## Model Format

The loader reads GGUF model files directly and memory-maps tensor data where possible. This keeps startup latency low and avoids eagerly copying model weights into Go-managed memory.

The CPU path supports the quantized tensor formats needed by the target DeepSeek-style sparse MoE models, including Q4_K_M, Q8_0, IQ2_XXS, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K variants.

## CPU Path

The CPU implementation uses:

- Go model loading and scheduler code.
- Architecture-specific SIMD kernels for hot quantized matrix operations.
- A mixture-of-experts execution path that routes tokens to selected experts.
- KV cache handling for autoregressive decode.

The default build keeps this path portable and avoids mandatory CGo dependencies.

## GPU Paths

CUDA and Vulkan are optional acceleration paths. They are selected at runtime with `-gpu` when a supported backend is available.

The experimental Metal backend is selected by building with `-tags metal` and setting `DS4_GPU_BACKEND=metal`. It uses CGo to call the local Metal bridge and is currently used for constrained-memory experiments on macOS.

## Code Provenance

Original baseline implementation by antirez:

- Repository: `https://github.com/antirez/ds4`
- First public commit: `b7bc7260d8f001dc32c02f923377049b8aa8ce2e`

This fork keeps the Go package/API shape while experimenting with GPU backends, model streaming, and memory-budgeted inference.

## Build

CPU build:

```bash
go build ./cmd/ds4chat
go build ./cmd/ds4server
```

CUDA/Vulkan-capable build:

```bash
go build ./cmd/ds4chat
./ds4chat -model /path/to/model.gguf -gpu
```

Metal-capable build on macOS:

```bash
CGO_ENABLED=1 go build -tags metal ./cmd/ds4chat
```

## Memory Budget Controls

The CPU and CUDA paths primarily depend on model quantization, KV cache size, and backend residency decisions.

The Metal path adds explicit environment controls for constrained-memory runs:

```bash
DS4_GPU_BACKEND=metal
DS4_METAL_ENABLE_MOE=1
DS4_METAL_STREAM_RAM_MB=24576
DS4_METAL_RESIDENT_HOT_MB=12288
DS4_METAL_STREAM_CACHE_RAM_MB=4096
DS4_METAL_COMPACT_EXPERT_CACHE_MB=8192
DS4_METAL_MEMORY_REPORT=1
```

These values describe the tested 24 GB envelope:

- 12 GB dense-first hot residency.
- 4 GB transient streamed tensor cache.
- 8 GB GPU-private compact expert cache.

See [metal.md](metal.md) for current measurements and caveats.

## Testing

Run package tests:

```bash
go test ./...
```

Run CUDA/Vulkan smoke tests with a local model:

```bash
./ds4chat -model /path/to/model.gguf -gpu -ctx 256 -n 1
```

Run a constrained Metal smoke test:

```bash
DS4_GPU_BACKEND=metal \
DS4_METAL_ENABLE_MOE=1 \
DS4_METAL_STREAM_RAM_MB=24576 \
DS4_METAL_RESIDENT_HOT_MB=12288 \
DS4_METAL_STREAM_CACHE_RAM_MB=4096 \
DS4_METAL_COMPACT_EXPERT_CACHE_MB=8192 \
DS4_METAL_MEMORY_REPORT=1 \
./ds4chat -model /path/to/model.gguf -ctx 256 -n 1 -temp 0 -gpu -fast
```

## Project Layout

```text
cmd/ds4chat/              Interactive CLI
cmd/ds4server/            OpenAI-compatible HTTP server
pkg/ds4/                  Go inference package
pkg/ds4/internal/simd/    CPU SIMD kernels
ds4_metal.*               Experimental Metal bridge
docs/                     Architecture, backend, and experiment notes
```

