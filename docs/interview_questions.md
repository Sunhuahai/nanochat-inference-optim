# Interview questions and answer points

## Measured result snapshot

The numbers below are medians from five repeats on an NVIDIA GeForce RTX 4060
8GB with PyTorch 2.9.1+cu128, nanochat commit `92d63d4`, the d14 base checkpoint,
greedy decoding, and 64 generated tokens.

| Prompt | Baseline TTFT | Fast-path TTFT | Fast-path change | Prefix-hit TTFT | Prefix-hit change | Peak VRAM (base / fast / prefix) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8.51 ms | 7.99 ms | -6.15% | 0.18 ms | -97.90% | 1168.9 / 1169.1 / 1160.2 MB |
| 512 | 15.23 ms | 14.98 ms | -1.61% | 0.41 ms | -97.30% | 1315.7 / 1319.2 / 1196.5 MB |
| 1024 | 25.75 ms | 25.26 ms | -1.91% | 0.72 ms | -97.19% | 1554.0 / 1557.1 / 1245.0 MB |
| 2048 | 48.06 ms | 46.72 ms | -2.79% | 1.35 ms | -97.19% | 2020.3 / 2023.3 / 1343.0 MB |

For the main 2048-token case, the complete measured row is:

| Engine | TTFT | TPOT | E2E | Output tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Upstream baseline | 48.06 ms | 6.416 ms | 452.28 ms | 141.51 | 2020.3 MB |
| Single-cache fast path | 46.72 ms | 6.413 ms | 450.89 ms | 141.94 | 2023.3 MB |
| Warm exact-prefix hit | 1.35 ms | 6.420 ms | 405.84 ms | 157.70 | 1343.0 MB |

Decode TPOT stayed approximately 6.1-6.4 ms across the three paths, which is
consistent with both optimizations targeting prefill rather than the decode
loop. Greedy output tokens matched the baseline for both optimized cold and
warm-prefix paths, and the unit-test suite passed (`3 passed`).

An interview-ready summary is:

> On an RTX 4060 with nanochat d14, reduced 2048-token prompt TTFT from 48.06
> to 46.72 ms (2.8%) by removing redundant single-sample KV copying, and to
> 1.35 ms (97.2%) on repeated exact-prefix hits, while preserving greedy token
> equivalence with the upstream engine.

## 1. What did you change in nanochat?

I made two deliberately small inference optimizations. First, for the common
`num_samples=1` path I prefill directly into the final decode KV cache instead
of allocating a temporary prompt cache and copying it. Second, I added a bounded
exact-prefix LRU cache that can reuse KV state and final prefill logits across
identical prompts.

## 2. Why is the single-cache path correct?

The prompt K/V written during prefill is exactly the state needed by subsequent
decode steps. The original two-cache design is useful when one prefill must be
replicated to multiple samples. With one sample, replication is unnecessary, so
writing prefill directly into the final cache preserves the same logical state.

## 3. What does it optimize?

It removes one prompt-sized KV allocation and one prompt-length KV copy. The
avoided work grows with prompt length, but measured latency can still be noisy.
The benefit should mainly appear in TTFT rather than decode TPOT. On the measured
2048-token workload, TTFT fell from 48.06 to 46.72 ms while TPOT was effectively
unchanged (6.416 vs 6.413 ms).

## 4. KV Cache memory formula?

For a decoder-only Transformer, ignoring allocator overhead:

`KV bytes ≈ 2 × L × T × H_kv × D_head × bytes_per_element × batch_size`

The factor 2 is K and V. GQA/MQA reduce `H_kv`, which directly reduces cache
memory and KV read traffic.

## 5. KV Cache vs Prefix Cache?

Normal KV Cache reuses history *within one autoregressive request*: during decode
we do not recompute the K/V of previous tokens. Prefix Cache reuses prefill state
*across requests* when their prefix is identical.

## 6. Why cache the final prefill logits too?

After prefill, the next token is sampled from the logits of the prompt's final
position. If I cached only K/V, I would still need extra model work to reconstruct
those logits. Caching them lets an exact-prefix hit skip the entire prefill.

## 7. Why do you cache `prev_embedding`?

Current nanochat has a smear operation that mixes the previous token embedding
into the current token. Therefore K/V is not the complete inference state.
`prev_embedding` must be restored too, otherwise a warm-prefix request may diverge
from a cold request.

## 8. Why only exact prefix matching?

It keeps correctness and implementation transparent. A production engine might
use block-level hashing or a radix tree for longest-prefix matching, but that
adds memory-management and eviction complexity that is unnecessary for this
small learning project.

## 9. Why does Prefix Cache mainly improve TTFT rather than TPOT?

It skips prefill. Once generation enters decode, both cached and uncached paths
execute the same one-token decode loop, so steady-state per-token latency should
be similar. For the 2048-token workload, a warm hit reduced TTFT from 48.06 to
1.35 ms, while TPOT was 6.420 ms versus the baseline's 6.416 ms.

## 10. Why must CUDA timing use synchronization?

CUDA launches kernels asynchronously. CPU wall-clock timing without a
synchronization point mostly measures enqueue time. Synchronizing before/after
the timed region makes the wall-clock interval include actual GPU work.

## 11. Why warm up before benchmarking?

The first iteration can include kernel loading, allocator activity, cache setup,
and autotuning. Warmup reduces this one-time noise so repeated measurements are
more representative.

## 12. Why use median results?

Latency measurements contain OS/runtime jitter. Median is less sensitive to
outliers than a simple mean for a small benchmark sample.

## 13. What are the limitations?

The prefix cache is exact-match only, GPU-resident, local to one Engine instance,
and has a fixed entry-count LRU rather than a byte budget. It is a learning
implementation, not a production serving cache. Also, the single-cache path did
not reduce measured peak allocated VRAM on this workload: at 2048 prompt tokens,
the baseline and fast path measured 2020.3 and 2023.3 MB. Avoiding a KV copy can
still reduce work without lowering the overall high-water mark when prefill
activations dominate it.

## 14. What would you do next?

The next reasonable extensions are a byte-budgeted cache, block-level longest
prefix matching, or continuous batching. I would only add them after the current
benchmark shows a clear bottleneck and correctness tests remain stable. Based on
the measurements, byte-budgeted eviction is the most direct next step because
the warm-prefix speedup is large and the cached KV state is GPU-resident.
