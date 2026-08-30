# Interview questions and answer points

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
benefit should mainly appear in TTFT and transient memory, and should scale with
prompt length. It should not materially change decode TPOT.

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
be similar.

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
implementation, not a production serving cache.

## 14. What would you do next?

The next reasonable extensions are a byte-budgeted cache, block-level longest
prefix matching, or continuous batching. I would only add them after the current
benchmark shows a clear bottleneck and correctness tests remain stable.
