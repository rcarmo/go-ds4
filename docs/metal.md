# Metal Experiments

The Metal backend is an experimental macOS path for studying whether large sparse MoE models can be run within a strict RAM envelope on smaller Macs. It borrows the practical shape of the `mlx-flash` work: memory-map the model, keep the most valuable weights resident, stream colder sections on demand, and cache compact expert views instead of materializing the whole model in memory.

## Hardware And Model

Current measurements were taken on:

- 36 GB M3 Max Mac.
- macOS Metal backend built with `CGO_ENABLED=1 go build -tags metal`.
- `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`, approximately 81 GB.
- Hard experimental budget: no more than 24 GB RAM.

This is intentionally below the model size. The goal is to test whether a sub-128 GB Mac can use streaming and residency planning to produce usable inference.

## Memory Split

The representative constrained split is:

```bash
DS4_GPU_BACKEND=metal
DS4_METAL_ENABLE_MOE=1
DS4_METAL_STREAM_RAM_MB=24576
DS4_METAL_RESIDENT_HOT_MB=12288
DS4_METAL_STREAM_CACHE_RAM_MB=4096
DS4_METAL_COMPACT_EXPERT_CACHE_MB=8192
DS4_METAL_MEMORY_REPORT=1
```

That budget is interpreted as:

| Region | Size | Purpose |
| --- | ---: | --- |
| Hot resident range | 12 GB | Dense-first model sections kept mapped/resident |
| Stream cache | 4 GB | Transient model windows for colder tensors |
| Compact expert cache | 8 GB | GPU-private compact views of routed experts |
| Total envelope | 24 GB | Upper bound for the experiment |

## Techniques

The Go Metal backend currently uses:

- CGo bridge into the local Metal host code.
- Q8_0 Metal matrix/vector kernels for supported tensors.
- Opt-in routed MoE acceleration with `DS4_METAL_ENABLE_MOE=1`.
- mmap-backed model views rather than an eager full-model load.
- Dense-first hot residency planning.
- A transient streamed tensor cache with eviction reporting.
- A GPU-private compact expert cache.
- A fresh-prompt prefill graph that keeps intermediate tensors device-resident where possible and reads back logits plus KV state.
- Batched shared-expert gate/up/SwiGLU work in the graph path.
- Corrected 4-wide router layout for `-fast` top-4 runs.
- An opt-in mapped routed gate/up pair kernel with route-weighted SwiGLU into the half-precision down RHS.
- An opt-in mapped routed down kernel that atomically accumulates selected expert outputs directly into `routedOut`, skipping the separate routed-down tensor sum.

The prefill graph is still incomplete compared with the ideal target. The remaining expensive section is the routed MoE path, especially route, compact, gather, and expert compute coordination.

## Representative Results

Token rates are measured from first token rather than process execution time.

| Path | Prompt | Prefill | Decode | Max RSS | Hot Resident | Stream Cache | Compact Expert Cache | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Metal graph, top-6 | 68 tok | 2.3 tok/s | 2.8 tok/s | 11.8 GiB | 11.75 GiB | 4 GiB cap, 333 evictions | 3.6 GiB live | Batched shared expert |
| Metal graph, top-4 `-fast` | 68 tok | 2.5 tok/s | 3.0 tok/s | 11.8 GiB | 11.75 GiB | 4 GiB cap, 294 evictions | 2.5 GiB live | Best current split |
| Metal graph, top-4 `-fast` | 132 tok | 1.9 tok/s | 1.9 tok/s | 11.7 GiB | 11.75 GiB | 4 GiB cap, 294 evictions | 2.5 GiB live | Longer prompt did not amortize better |
| Metal graph, top-4 `-fast` | 84 tok | 3.3 tok/s | 3.7 tok/s | 11.8 GiB | 11.75 GiB | 4 GiB cap, 358 evictions | 3.4 GiB live | Default mapped gate, mapped up, fused activation |
| Metal graph, top-4 `-fast`, pair SwiGLU | 84 tok | 3.1 tok/s | 3.4 tok/s | 11.7 GiB | 11.75 GiB | 4 GiB cap, 358 evictions | 3.4 GiB live | Opt-in fused gate/up pair regressed, not default |
| Metal graph, top-4 `-fast`, down+sum fusion | 84 tok | 3.1 tok/s | 3.2 tok/s | 11.7 GiB | 11.75 GiB | 4 GiB cap, 358 evictions | 3.4 GiB live | Opt-in atomic down accumulation; within run noise, not default |

## Optional Switches

These switches remain useful for experiments but are not default wins:

```bash
DS4_METAL_MOE_PREFILL_COMPACT=1
```

Forces synchronized compact-expert prefill. It regressed the 68-token test to roughly 1.7 tok/s.

```bash
DS4_METAL_MOE_PREFILL_MV=1
```

Forces routed matrix/vector work instead of mapped matrix/matrix work. It measured around 2.0 tok/s on the 68-token top-4 test, so the mapped MM path remains the default.

```bash
DS4_METAL_ROUTED_MM_PAIR_SWIGLU_FUSION=1
```

Enables the experimental mapped routed gate/up pair kernel. It computes gate and up in one dispatch and writes route-weighted SwiGLU directly into the half-precision down RHS. On the 84-token top-4 test it regressed from 3.3 tok/s to 3.1 tok/s, so it is not enabled by default.

```bash
DS4_METAL_ROUTED_MM_DOWN_SUM_FUSION=1
```

Enables the experimental mapped routed down+sum kernel. It zeros `routedOut`, then has each selected expert's down projection atomically accumulate into the token-major output. This removes the separate routed-down sum kernels without adding a command-buffer boundary, but the atomic writes made the result noisy rather than clearly faster in the 84-token top-4 test.

## Current Conclusions

The constrained Metal path can stay inside the 24 GB envelope on a 36 GB M3 Max Mac, which validates the memory approach. The speed is not yet good enough for a practical chat experience.

The current bottleneck is not raw RSS. The bigger issues are routed MoE overhead, compact/gather traffic, synchronization around streamed views, and not enough GPU-side fusion across route, gather, expert compute, and scatter. Increasing resident size helps avoid evictions, but the best measured split still leaves performance dominated by the routed expert path.

Normal prefill already runs most layer work inside one Metal command batch; profiling modes intentionally break that batch to time individual stages. The remaining boundary work is mostly about reducing per-stage encoders and intermediate memory traffic inside that batch rather than eliminating many full command-buffer commits.
