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

Use the same Python environment as upstream nanochat.

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
pip install -e .

# clone this repository next to it or anywhere in the same environment
pip install -e /path/to/nanochat-inference-optim
```

Upstream nanochat currently requires Python >=3.10 and provides its own PyTorch
and GPU dependency setup. Follow its README for checkpoint preparation.

## Run unit tests

The standalone prefix-cache tests do not require a nanochat checkpoint:

```bash
pytest -q
```

## Check correctness against upstream

After a nanochat base checkpoint is available:

```bash
python benchmarks/check_correctness.py
```

This uses greedy decoding and requires both the cold optimized path and a warm
prefix-cache hit to exactly match upstream output tokens.

## Run benchmark

After a nanochat base checkpoint is available:

```bash
python benchmarks/bench_inference.py \
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

- single-cache fast path: benefit grows with prompt length; decode TPOT is nearly
  unchanged;
- exact Prefix Cache: large TTFT reduction for repeated long prompts, at the cost
  of keeping cached KV tensors in GPU memory.

Record real results in the table below after running on your GPU:

| Prompt | Baseline TTFT | Fast-path TTFT | Prefix-hit TTFT | Peak memory change |
|---:|---:|---:|---:|---:|
| 128 | TBD | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD | TBD |
| 2048 | TBD | TBD | TBD | TBD |

## Resume wording after benchmark

A conservative version before inserting real numbers:

> Analyzed nanochat's autoregressive inference path and built a TTFT/TPOT/
> throughput/VRAM benchmark. Implemented a single-sample KV-cache fast path to
> remove redundant prefill-to-decode KV copying, and an exact-prefix LRU cache
> that reuses KV, final prefill logits, and nanochat smear state across repeated
> prompts; validated correctness with greedy decoding and ablation benchmarks.

Once measured, replace the qualitative statement with your actual TTFT / memory
numbers.

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
