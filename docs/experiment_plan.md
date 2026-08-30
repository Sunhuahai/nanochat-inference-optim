# Experiment plan

The project intentionally tests only two small changes. Do not claim a speedup
before measuring it on the target GPU.

## A. Single-sample KV fast path

Question: does avoiding the temporary prompt-sized cache and the prompt-KV copy
reduce TTFT / peak transient memory for `num_samples=1`?

Compare:

- `nanochat.engine.Engine`
- `OptimizedEngine(prefix_cache_entries=0)`

Workloads:

| Prompt tokens | New tokens | Temperature |
|---:|---:|---:|
| 128 | 64 | 0 |
| 512 | 64 | 0 |
| 1024 | 64 | 0 |
| 2048 | 64 | 0 |

Primary metrics: TTFT, E2E latency, peak allocated CUDA memory.

Expected behavior: the gain should become easier to see as the prompt grows,
because the avoided KV copy is proportional to prompt length. Decode TPOT should
remain almost unchanged because the decode loop itself is unchanged.

## B. Exact Prefix KV Cache

Question: how much TTFT can be saved when the exact same long prompt is reused?

For each prompt length:

1. clear prefix cache;
2. run one cold request to populate it;
3. run repeated identical requests;
4. compare warm-cache TTFT to baseline TTFT.

Primary metric: TTFT. Secondary metrics: cache memory cost and hit rate.

Expected behavior: the cache trades extra GPU memory for avoiding prefill compute.
It should mainly improve TTFT, not steady-state TPOT.

## Correctness checks

Use greedy decoding (`temperature=0`) and compare token sequences from the
reference and optimized engines. For prefix-cache tests, compare both cold and
warm paths.

The benchmark script uses CUDA synchronization around timing boundaries because
CUDA kernel launches are asynchronous.
