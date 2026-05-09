# ds4.c Streaming Experiment

This branch is an experimental C/Metal-only attempt to make DeepSeek V4 Flash
inference usable on Macs with less than 128 GB of unified memory.

The original `ds4.c` path assumes a machine class where the q2 GGUF can fit in
memory comfortably. This branch asks a narrower question: can we use the
`mlx-flash` style of file-backed model access, Metal buffers over mmap-backed
views, and a small set of GPU-private caches to keep enough of the model hot for
interactive inference on smaller Macs?

Our answer so far is: technically yes, but not practically on this machine. The
streaming path runs and stays inside a 24 GiB test envelope, but generation is
still far below usable interactive speed. The best local first-token-basis
decode result is about `0.83 tok/s`, which is useful as a proof of direction,
not as a comfortable local model.

## Test System

These notes were produced on:

- Machine: MacBook Pro `Mac15,10`
- Chip: Apple M3 Max
- CPU: 14 cores, 10 performance and 4 efficiency
- GPU: 30-core Apple GPU
- Memory: 36 GB unified memory
- OS: macOS 26.4.1, build 25E253
- Kernel: Darwin 25.4.0 arm64

This is well below the 128 GB configuration normally expected for the q2 model.
That is the point of the experiment: if this approach cannot become useful here,
it is unlikely to be a satisfying path for the smallest target machines without
deeper changes.

## Rationale

The working idea was to split model access into three pieces:

- Always-hot mmap-backed Metal views for frequently used non-expert tensors.
- A transient stream-view cache for other file-backed model ranges.
- A GPU-private compact expert cache for routed MoE expert slices.

This follows the spirit of `mlx-flash`: avoid treating the whole model as one
resident allocation, lean on mmap and demand paging, and keep the active working
set small enough for lower-memory Macs. The C version keeps the existing
DeepSeek V4 Flash-specific graph executor and adds streaming weight views around
the routed MoE and other model accesses.

The core constraint is routed MoE. Every decoded token selects experts, gathers
their gate/up/down tensors, and runs the expert matvecs. Under streaming, that
route/compact/gather path dominates. The current best path avoids full-expert
views and copies cold expert misses directly from the source mmap view into the
compact buffer while also filling the GPU-private per-expert cache.

## Current Settings

The conservative constrained test is:

```sh
make test-constrained-ram
```

with a 16 GiB streaming envelope:

- `RAM_TEST_MB=16384`
- `RAM_TEST_RESIDENT_HOT_MB=8192`
- `RAM_TEST_STREAM_CACHE_RAM_MB=4096`
- `RAM_TEST_COMPACT_CACHE_MB=4096`
- `RAM_TEST_STREAM_CACHE=32`
- `RAM_TEST_STREAM_WINDOW_MB=8`
- `RAM_TEST_PIN_MAX_MB=1`

The largest local envelope we currently allow is:

```sh
make test-constrained-ram-24gb
```

with:

- `RAM_TEST_MB=24576`
- `RAM_TEST_RESIDENT_HOT_MB=8192`
- `RAM_TEST_STREAM_CACHE_RAM_MB=8192`
- `RAM_TEST_COMPACT_CACHE_MB=8192`

That gives an even `8 GiB hot + 8 GiB stream + 8 GiB compact` split. Anything
above 24 GiB is intentionally out of scope for this machine because it leaves
too little headroom for the OS and the rest of the process.

## Running

Build:

```sh
make ds4
```

Run with streamed weights:

```sh
./ds4 --stream-weights --nothink --ctx 256 --tokens 8 -p 'Hi'
```

The main environment knobs are:

- `DS4_METAL_STREAM_WEIGHTS=1`: enable streamed model weights.
- `DS4_METAL_STREAM_RAM_MB`: total streaming envelope.
- `DS4_METAL_RESIDENT_HOT_MB`: planned always-hot model ranges.
- `DS4_METAL_STREAM_CACHE_RAM_MB`: transient mmap-backed stream-view cache.
- `DS4_METAL_COMPACT_EXPERT_CACHE_MB`: GPU-private routed expert slice cache.
- `DS4_METAL_STREAM_CACHE`: stream-view slot count.
- `DS4_METAL_STREAM_WINDOW_MB`: stream-view window size.
- `DS4_METAL_RESIDENCY_PLAN_DUMP=1`: print the selected hot ranges.

## Results

Representative first-token-basis results:

| Run | RAM MiB | Hot MiB | Stream MiB | Compact MiB | Tokens | Gen tok/s | Max RSS GiB | Stream Evict | Compact Evict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Early no-hot baseline | 16384 | 0 | 3849 peak | 4096 | 8 | 0.37 | 4.29 | 7767 | 1024 |
| 16 GiB default | 16384 | 8192 | 4096 | 4096 | 8 | 0.74 | 7.72 | 2871 | 1024 |
| 16 GiB more compact | 16384 | 8192 | 2048 | 6144 | 8 | 0.77 | 7.72 | 2751 | 0 |
| 24 GiB balanced | 24576 | 9216 | 7168 | 8192 | 8 | 0.77 | 8.25 | 2743 | 0 |
| 24 GiB stream-heavy | 24576 | 9216 | 9216 | 6144 | 8 | 0.72 | 8.25 | 2743 | 0 |
| 24 GiB optimized | 24576 | 8192 | 8192 | 8192 | 8 | 0.83 | 7.72 | 2751 | 0 |
| 24 GiB optimized, longer check | 24576 | 8192 | 8192 | 8192 | 16 | 0.82 | 7.72 | 3116 | 0 |

The raw matrix is in `benchmarks/streaming_matrix.csv`, with the summary in
`benchmarks/streaming_matrix.md`.

## Conclusions

Planned hot residency is the largest confirmed win. Moving frequently used
non-expert tensors into hot mmap-backed Metal views raised the 8-token
first-token-basis run from about `0.37 tok/s` to `0.74 tok/s`.

The compact expert cache is useful up to the active expert working set. A 4 GiB
compact cache still evicts; 6 GiB removes compact evictions for the short test;
8 GiB is the safer 24 GiB setting because the longer check reaches about
6.79 GiB compact live.

The stream-view cache remains the main churn signal. Larger stream budgets and
more stream slots did not turn into meaningful reuse on this prompt; final
stream evictions remained in the thousands. The remaining cost is repeated
per-token wrapping, blit/gather work, and page churn around ranges that do not
fit the current cache shape well.

The current 24 GiB ceiling is safe on the test system, but it is not fast enough
to make this configuration a viable daily inference setup. The best measured
split is `8 GiB hot + 8 GiB stream + 8 GiB compact`, reaching about `0.83 tok/s`.

## What Did Not Pan Out

We tried a deeper direct selected-expert compute path that avoided compact
gate/up/down buffers and used the six selected per-expert cache/source slices
directly. It avoided full-expert views, but the extra per-expert dispatch fanout
was slower than the compact gather path, measuring about `0.78 tok/s` under the
same 24 GiB envelope. That code was removed.

The next plausible path would need true GPU-side gather/compute fusion: one or
two kernels that consume the selected expert slices without full-expert views,
without compact buffers, and without multiplying dispatch count by six. That is
deeper than the current branch implements.

## Status

This branch is a record of the streaming experiment, not a recommendation for
production inference on this hardware. It shows that sub-128 GB execution is
possible in a constrained C/Metal implementation, but not yet practical on our
36 GB M3 Max configuration.
