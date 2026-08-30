# nanochat-inference-optim

Small, interview-friendly inference optimizations for
[karpathy/nanochat](https://github.com/karpathy/nanochat).

The goal is **not** to reimplement vLLM. This repository focuses on two small
changes that are easy to understand, verify, and benchmark end to end.

The implementation is aligned with upstream nanochat `master` at commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (checked 2026-08-30).

## What is optimized?

### 1. Single-sample KV Cache fast path

Current nanochat `Engine.generate()` performs a batch-1 prefill into a temporary
KV cache and then allocates a decode cache and copies the prompt KV into it.
That design is useful when one prompt is expanded to multiple samples, but for
`num_samples=1` the copy is unnecessary.

This project instead does:

```text
Reference, num_samples=1
prompt -> temporary prefill KV -> copy -> final decode KV -> decode

Optimized
prompt -----------------------> final decode KV -> decode
```

This removes one prompt-sized KV allocation and one prompt-length KV copy.
Expected impact: mostly **TTFT** and transient memory; little change to decode
TPOT.

### 2. Exact Prefix KV Cache

When a token-identical prompt is repeated, the optimized engine can restore its
prefill state instead of recomputing it.

```text
prompt tokens
    |
    v
BLAKE2 hash -> LRU lookup ---- hit ----> KV + logits + smear state
                    |
                   miss
                    v
                 prefill
```

The cache stores:

- K cache
- V cache
- final prefill logits
- `prev_embedding` used by nanochat's smear operation

It is deliberately **exact-match only**. No radix tree, paging, custom CUDA, or
continuous batching is required.

## Why this project is useful for interviews

The code is small enough to explain line by line, but naturally connects to:

- KV Cache memory calculation
- GQA and KV memory reduction
- Prefill vs Decode
- TTFT vs TPOT
- GPU asynchronous timing
- Prefix Cache correctness
- cache-memory / latency trade-offs

See [`docs/interview_questions.md`](docs/interview_questions.md) for the expected
follow-up questions.

## Repository structure

```text
nanochat_optim/
  engine.py          # optimized Engine
  prefix_cache.py    # exact prompt LRU cache
benchmarks/
  bench_inference.py  # TTFT / TPOT / throughput / peak-memory benchmark
  check_correctness.py # greedy cold/warm equivalence check
tests/
  test_prefix_cache.py
docs/
  experiment_plan.md
  interview_questions.md
```

## Setup

Dependencies are isolated in a repository-local `.venv` and locked with `uv`.
The setup also creates a project-local checkout of the tested upstream nanochat
commit under `.upstream/`; it does not install anything into the system Python.

```bash
bash scripts/setup_env.sh
```

`uv` reuses its global download cache across projects, while each project keeps
its own installed environment and upstream source. The wrapper
`scripts/run_env.sh` selects both when running commands. Model artifacts follow
nanochat's standard layout under `~/.cache/nanochat`; a base checkpoint therefore
looks like `~/.cache/nanochat/base_checkpoints/d14/model_002192.pt`.

## Run unit tests

The standalone prefix-cache tests do not require a nanochat checkpoint:

```bash
scripts/run_env.sh pytest -q
```

## Check correctness against upstream

After a nanochat base checkpoint is available:

```bash
scripts/run_env.sh python benchmarks/check_correctness.py
```

This uses greedy decoding and requires both the cold optimized path and a warm
prefix-cache hit to exactly match upstream output tokens.

## Run benchmark

After a nanochat base checkpoint is available:

```bash
scripts/run_env.sh python benchmarks/bench_inference.py \
  --prompt-lengths 128 512 1024 2048 \
  --max-new-tokens 64 \
  --repeats 5 \
  --output results/bench.csv
```

The benchmark compares:

1. upstream `Engine`;
2. optimized single-cache path with prefix cache disabled;
3. warm exact-prefix cache hits.

It reports median:

- TTFT (time to first token)
- TPOT (time per output token after the first)
- E2E latency
- output tokens/s
- peak CUDA memory allocated

## What results should I expect?

Do **not** hard-code a speedup into a resume before measurement.

Expected trends:

- single-cache fast path: avoided KV-copy work grows with prompt length, though
  measured latency can be noisy; decode TPOT is nearly unchanged;
- exact Prefix Cache: large TTFT reduction for repeated long prompts, at the cost
  of keeping cached KV tensors in GPU memory.

Measured on an NVIDIA GeForce RTX 4060 8GB with PyTorch 2.9.1+cu128,
nanochat d14, greedy decoding, 64 generated tokens, and five repeats (median):

| Prompt | Baseline TTFT | Fast-path TTFT | Prefix-hit TTFT | Peak VRAM (base / fast / prefix) |
|---:|---:|---:|---:|---:|
| 128 | 8.51 ms | 7.99 ms (-6.15%) | 0.18 ms (-97.90%) | 1168.9 / 1169.1 / 1160.2 MB |
| 512 | 15.23 ms | 14.98 ms (-1.61%) | 0.41 ms (-97.30%) | 1315.7 / 1319.2 / 1196.5 MB |
| 1024 | 25.75 ms | 25.26 ms (-1.91%) | 0.72 ms (-97.19%) | 1554.0 / 1557.1 / 1245.0 MB |
| 2048 | 48.06 ms | 46.72 ms (-2.79%) | 1.35 ms (-97.19%) | 2020.3 / 2023.3 / 1343.0 MB |

Decode TPOT remained approximately 6.1-6.4 ms. The single-cache path reduced
TTFT but did not reduce the measured peak-memory high-water mark on this setup;
the Prefix Cache result is specifically for repeated, token-identical prompts.

## Resume wording after benchmark

> Analyzed nanochat's autoregressive inference path and built a TTFT/TPOT/
> throughput/VRAM benchmark. On RTX 4060 with nanochat d14, reduced 2048-token
> prompt TTFT from 48.06 to 46.72 ms (2.8%) by removing redundant single-sample
> KV copying, and to 1.35 ms (97.2%) on exact-prefix hits; preserved greedy-token
> equivalence across the upstream, optimized cold, and warm-cache paths.

## Scope

This is intentionally not a production serving engine. Out of scope:

- PagedAttention
- continuous batching
- distributed inference
- custom CUDA kernels
- CPU KV swapping
- speculative decoding

Those are good follow-up topics, but they would make this learning project much
larger than necessary.

## Attribution

This project is an independent educational extension of nanochat. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
