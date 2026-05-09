# go-ds4

![go-ds4](docs/icon-256.png)

A Go inference engine for large sparse MoE language models such as DeepSeek V4 Flash, with CPU, CUDA, Vulkan, and experimental Metal paths.

The default build remains a self-contained Go implementation with no mandatory C/C++ runtime. GPU paths are optional. The CUDA/Vulkan backends target Linux and discrete GPUs; the Metal backend is an experimental macOS path for studying whether mmap-backed streaming, hot residency, and compact expert caches can make large MoE models usable on Macs with less than 128 GB of RAM.

## Status

The most mature path is the Go CPU/CUDA implementation. The Metal backend currently runs under a constrained 24 GB memory budget on a 36 GB M3 Max Mac, but it is still a research path rather than a practical interactive backend.

Current representative numbers:

| Backend | Hardware | Prompt | Prefill | Decode | Memory |
| --- | --- | ---: | ---: | ---: | ---: |
| CUDA, batched + shared memory | i7-12700 + RTX 3060 12 GB | decode | n/a | 2.09 tok/s | ~8 GB VRAM |
| CPU AVX2, top-4 `-fast` | i7-12700 | decode | n/a | 1.66 tok/s | system RAM |
| CPU AVX2, top-6 | i7-12700 | decode | n/a | 1.09 tok/s | system RAM |
| Metal graph, top-4 `-fast` | 36 GB M3 Max Mac | 84 tokens | 3.3 tok/s | 3.7 tok/s | 11.8 GiB RSS, 24 GB cap |
| Metal graph, top-4 `-fast` | 36 GB M3 Max Mac | 68 tokens | 2.5 tok/s | 3.0 tok/s | 11.8 GiB RSS, 24 GB cap |
| Metal graph, top-4 `-fast` | 36 GB M3 Max Mac | 132 tokens | 1.9 tok/s | 1.9 tok/s | 11.7 GiB RSS, 24 GB cap |

See [docs/metal.md](docs/metal.md) for the Metal experiment notes, memory split, and current bottlenecks.

## Quick Start

Build and run the CLI:

```bash
go build ./cmd/ds4chat
./ds4chat -model /path/to/model.gguf
```

Run the OpenAI-compatible server:

```bash
go run ./cmd/ds4server -model /path/to/model.gguf -listen :8080
```

Enable CUDA or Vulkan:

```bash
./ds4chat -model /path/to/model.gguf -gpu
```

Build the experimental Metal backend on macOS:

```bash
CGO_ENABLED=1 go build -tags metal ./cmd/ds4chat
DS4_GPU_BACKEND=metal DS4_METAL_ENABLE_MOE=1 ./ds4chat -model /path/to/model.gguf -ctx 256 -n 1 -temp 0 -gpu
```

## Documentation

- [API and server usage](docs/api.md)
- [Implementation notes](docs/implementation.md)
- [Architecture](docs/architecture.md)
- [GPU backends](docs/gpu.md)
- [Metal experiments](docs/metal.md)
- [Optimization history](docs/optimization.md)
- [SIMD kernels](docs/simd.md)

## Branches

- `main`: Go implementation, including the experimental Metal backend.
- `c/streaming`: experimental C implementation using mlx-flash-style streaming, hot residency, and compact expert caching.

## Motivation

In December 2025, China will have an already-proven open model architecture that they can train for $280K while Nvidia charges $16K for a middling GPU. The strategic question is not just model quality; it is whether inference can be made practical on commodity hardware.

This repository is a playground for that question:

- Write the inference path in Go where possible.
- Keep the CPU implementation understandable and portable.
- Use GPU acceleration where it meaningfully improves throughput.
- Study memory-mapped and streaming execution so large sparse models can run on smaller machines.

The goal is to move useful implementation ideas between the Go and C paths while keeping each branch easy to inspect.

## License

MIT, same as upstream [antirez/ds4](https://github.com/antirez/ds4).

## Acknowledgments

Thanks to [antirez](https://github.com/antirez) for the original DS4 C implementation, DeepSeek for the model architecture and weights, and the Go project for making portable deployment practical.
