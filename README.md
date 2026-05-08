# go-ds4

![go-ds4](docs/icon-256.png)

A pure Go inference engine for [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V3), ported from [@antirez's excellent single-file C implementation](https://github.com/antirez/ds4).

**Single static binary. AVX2 + CUDA PTX. 1.27 tok/s on CPU, 5.5× faster on GPU.**

| Mode | tok/s (decode) | Hardware | Notes |
|---|---|---|---|
| **CPU AVX2** (top-6) | **1.09** | i7-12700, 6 cores | Pure Go + hand-written assembly |
| **CPU AVX2** (top-4 fast) | **1.27** | i7-12700, 6 cores | `FastExperts` mode |
| **GPU CUDA** Q8_0 GEMV | **5.5×** faster | RTX 3060 12GB | Pure Go, no CGo |

The default build produces a **fully self-contained static binary** with no C dependencies. All hot paths use hand-written SIMD assembly (AVX2+FMA on amd64, NEON stubs on arm64). GPU acceleration via CUDA PTX is opt-in and loaded at runtime via `purego` (no CGo, no CUDA toolkit dependency).

## Quick Start

```bash
# Download model (~86 GB Q2 quantization)
./download_model.sh

# Run inference
go run ./cmd/ds4gen/ gguf/ds4-q2.gguf "Hello, world"

# Run OpenAI-compatible server
go run ./cmd/ds4server/ -model gguf/ds4-q2.gguf -listen :8080 -fast
```

## API

```go
import "github.com/rcarmo/go-ds4/ds4"

engine, _ := ds4.OpenEngineWithOptions(ds4.EngineOptions{
    ModelPath:   "gguf/ds4-q2.gguf",
    FastExperts: true,  // top-4 mode (+17% tok/s)
    UseGPU:      true,  // CUDA/Vulkan if available
})
defer engine.Close()

session := engine.NewSession(4096)
tokens := engine.Vocab.EncodeChatPrompt("", "Why is the sky blue?", false)
for _, t := range tokens {
    session.Eval(t)
}

// Autoregressive generation
for i := 0; i < 256; i++ {
    tok := ds4.Sample(session.Logits, 0.7, 40, 0.9, 0)
    if tok == engine.Vocab.EOS { break }
    fmt.Print(engine.Vocab.TokenText(tok))
    session.Eval(tok)
}
```

## OpenAI-Compatible Server

```bash
go build ./cmd/ds4server/
./ds4server -model gguf/ds4-q2.gguf -listen :8080 -fast

# Use with any OpenAI client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Supports: SSE streaming, temperature/top-k/top-p sampling, usage stats, client disconnect detection.

## Architecture

### Model: DeepSeek V4 Flash

- **43 layers**, 4096-dim embeddings, 64 attention heads
- **MLA** (Multi-head Latent Attention): 1 KV head, LoRA Q/O projections
- **256 routed experts** (IQ2_XXS/Q2_K) + 1 shared expert (Q8_0) per layer
- **Hyper-connections**: 4 parallel residual streams with Sinkhorn-normalized mixing
- **KV compression**: ratio-4 (with indexer) and ratio-128 layers
- **128 GB** model (Q2 quantization), served via mmap

### Inference Pipeline

```
Token → F16 Embed → [43 layers × (HC-pre → Attn → HC-post → HC-pre → MoE → HC-post)] → HC collapse → RMSNorm → Output Q8_0 → Logits
```

Each layer:
1. **HC pre**: Sinkhorn-normalized stream mixing (learned 4×4 combine matrix)
2. **MLA attention**: Q LoRA (4096→1024→32768), KV projection, SWA + compressed KV scoring
3. **MoE FFN**: top-6 expert routing (IQ2_XXS gate/up, Q2_K down) + shared expert (Q8_0)
4. **HC post**: residual injection with learned post-weights

## Performance

### Kernel Benchmarks (i7-12700)

| Kernel | Dimension | Time | Implementation |
|---|---|---|---|
| DotQ8_0F32 | 4096 | 572 ns | AVX2 VCVTPH2PS + VPMOVSXBD + FMA |
| VecDotIQ2XXSQ8K | 4096 | 807 ns | AVX2 DotIQ2Group32 (zero-copy grid) |
| VecDotQ2KQ8K | 4096 | 879 ns | AVX2 DotQ2Group16 (VPAND+VPMADDUBSW) |
| DotI8 | 32 | 6 ns | AVX2 VPMADDUBSW + VPMADDWD |

### GPU Benchmarks (RTX 3060, pure Go dispatch)

| Operation | Dimension | GPU | CPU | Speedup |
|---|---|---|---|---|
| Q8_0 GEMV | 4096→1024 | 110 µs | 586 µs | **5.3×** |
| Q8_0 GEMV | 4096→32768 | 3.4 ms | 18.7 ms | **5.5×** |

### End-to-End Decode Profile

| Function | CPU% | What |
|---|---|---|
| `simd.DotQ8_0F32` | 40% | Q8_0 attention/output projections (AVX2) |
| `VecDotIQ2XXSQ8K_SIMD` | 21% | IQ2_XXS expert gate/up (AVX2 grid lookup) |
| `simd.DotIQ2Group32` | 13% | IQ2 inner int8×int8 dot (AVX2) |
| `vecDotQ2KQ8K_scalar` | 13% | Q2_K expert down (AVX2 bit extraction) |
| `simd.DotQ2Group16` | 8% | Q2K inner dot (AVX2 VPMADDUBSW) |

**95% of CPU time is in hand-written AVX2 SIMD assembly.**

## Code Provenance

### From [antirez/ds4](https://github.com/antirez/ds4)
- Model architecture, constants, tensor layout, tokenizer design
- Hyper-connection Sinkhorn split logic
- KV compressor/indexer decode path
- Attention sink initialization
- Chat template encoding

### From [rcarmo/go-pherence](https://github.com/rcarmo/go-pherence)
- `ds4/simd/` package: AVX2+FMA and NEON assembly for Sdot, Saxpy, RMSNorm, VecSiLUMul
- `ds4/gpu/` Vulkan backend: purego dlopen, buffer management, SPIR-V dispatch pipeline
- `ds4/gpu/` CUDA backend: purego dlopen libcuda.so.1, PTX module load, kernel launch

### Original to this project
- DS4 Q8_0 34-byte block AVX2 kernel (VCVTPH2PS F16 scale + VPMOVSXBD + FMA)
- DS4 IQ2_XXS zero-copy grid lookup kernel (VMOVQ + VINSERTI128 + VPMADDUBSW)
- DS4 Q2_K bit-extraction kernel (VPAND + VPSRLW + VPUNPCKLBW + VPMADDUBSW)
- CUDA PTX Q8_0 GEMV kernel (per-row cooperative reduction, F16 native decode)
- GLSL/SPIR-V Q8_0 GEMV shader (Vulkan compute, software F16 decode)
- Logical ring KV cache (raw + compressed + indexer, no memmove)
- FastExperts mode (top-4 selection with weight renormalization)
- Session save/load v4 format with ring cursor persistence
- OpenAI-compatible HTTP server with SSE streaming

## Build

```bash
# Pure Go (default) — single static binary
CGO_ENABLED=0 go build ./cmd/ds4gen/
CGO_ENABLED=0 go build ./cmd/ds4server/

# With GPU support (still no CGo — GPU loaded at runtime)
go build -tags gpu ./cmd/ds4server/
```

## Memory Budget

The 86 GB model is served via mmap with configurable memory control:

| Mode | RSS | How |
|---|---|---|
| Full mmap (default) | ~7 GB | Demand-page only active expert pages |
| PinNonExpert | ~6.5 GB locked | mlock dense weights, MADV_DONTNEED cold experts |
| DiskStreaming | ~2 GB | Read expert weights from NVMe on demand |

## Testing

```bash
go test ./ds4/...
go test ./ds4/gpu/...
```

## Project Structure

```
cmd/ds4gen/       — CLI text generation
cmd/ds4server/    — OpenAI-compatible HTTP server
ds4/              — Core inference engine
  attention.go    — MLA attention + compressor + indexer
  compressor.go   — KV compression decode path
  gpu_bridge.go   — GPU dispatch bridge (CUDA/Vulkan)
  hc.go           — Hyper-connection Sinkhorn split
  kvcache.go      — Logical ring KV cache (raw + compressed)
  moe.go          — MoE routing + parallel expert execution
  ops.go          — Matmul, RMSNorm, RoPE, softmax, SiLU
  quant.go        — Q8_0, Q2_K, IQ2_XXS dot products
  session.go      — Engine, session lifecycle, sampling
  shape.go        — Model constants (43 layers, 256 experts, etc.)
  tokenizer.go    — GPT-2 BPE with JoyAI pre-tokenizer
  simd/           — Hand-written AVX2 + NEON assembly kernels
  gpu/            — Vulkan SPIR-V + CUDA PTX compute backends
```

## License

MIT — same as upstream [antirez/ds4](https://github.com/antirez/ds4).

## Acknowledgments

- [@antirez](https://github.com/antirez) for the original ds4 C implementation — a masterclass in minimal LLM inference
- DeepSeek for the V4 Flash architecture and weights
- The Go team for making assembly-friendly compiled binaries trivial to deploy
