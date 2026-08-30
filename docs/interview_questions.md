# 面试问题与回答要点

## 实测结果速览

以下数据在 NVIDIA GeForce RTX 4060 8GB、PyTorch 2.9.1+cu128、nanochat
提交 `92d63d4`、d14 base checkpoint、greedy decoding、生成 64 个 token 的
条件下测得，每项重复 5 次并取中位数。

| Prompt | Baseline TTFT | 快速路径 TTFT | 快速路径变化 | Prefix 命中 TTFT | Prefix 命中变化 | 峰值显存（base / fast / prefix） |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8.51 ms | 7.99 ms | -6.15% | 0.18 ms | -97.90% | 1168.9 / 1169.1 / 1160.2 MB |
| 512 | 15.23 ms | 14.98 ms | -1.61% | 0.41 ms | -97.30% | 1315.7 / 1319.2 / 1196.5 MB |
| 1024 | 25.75 ms | 25.26 ms | -1.91% | 0.72 ms | -97.19% | 1554.0 / 1557.1 / 1245.0 MB |
| 2048 | 48.06 ms | 46.72 ms | -2.79% | 1.35 ms | -97.19% | 2020.3 / 2023.3 / 1343.0 MB |

核心的 2048-token 场景完整数据如下：

| Engine | TTFT | TPOT | E2E | 输出 tok/s | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| 上游 baseline | 48.06 ms | 6.416 ms | 452.28 ms | 141.51 | 2020.3 MB |
| 单 cache 快速路径 | 46.72 ms | 6.413 ms | 450.89 ms | 141.94 | 2023.3 MB |
| Warm 精确 Prefix 命中 | 1.35 ms | 6.420 ms | 405.84 ms | 157.70 | 1343.0 MB |

三条路径的 decode TPOT 都保持在约 6.1-6.4 ms，符合两个优化都针对 prefill、
没有改变 decode loop 的预期。优化后的 cold path 和 warm Prefix path 均与 baseline
生成完全一致的 greedy token，单元测试结果为 `3 passed`。

面试中可以概括为：

> 在 RTX 4060、nanochat d14 上，通过移除冗余的单样本 KV 复制，将 2048-token
> prompt 的 TTFT 从 48.06 ms 降至 46.72 ms（2.8%）；通过精确 Prefix Cache，
> 将重复 prompt 的 TTFT 降至 1.35 ms（97.2%），并保持与上游引擎的 greedy token
> 完全一致。

## 1. 你对 nanochat 做了什么改动？

我实现了两个刻意保持小而清晰的推理优化。第一，对于常见的 `num_samples=1` 场景，
prefill 直接写入最终 decode KV Cache，不再先分配临时 prompt cache 再复制。第二，
加入有界的精确 Prefix LRU Cache，使 token 序列完全相同的请求可以复用 KV 状态和
prefill 最后一个位置的 logits。

## 2. 为什么单 cache 路径在逻辑上是正确的？

Prefill 写入的 prompt K/V 正是后续 decode 所需要的状态。原始双 cache 设计的价值在于
一次 prefill 可以复制到多个采样分支；只有一个样本时无需复制。因此，直接写入最终
cache 不会改变推理状态，只是消除了中间存储和复制。

## 3. 单 cache 路径具体优化了什么？

它省去一个 prompt 大小的 KV 分配，以及一次与 prompt 长度成正比的 KV 复制。省去的
工作量会随 prompt 变长而增加，但真实延迟仍会有测量噪声。收益应主要体现在 TTFT，
而不是 decode TPOT。在 2048-token 实测中，TTFT 从 48.06 ms 降至 46.72 ms，
TPOT 则基本不变（6.416 ms 对 6.413 ms）。

## 4. KV Cache 的显存公式是什么？

对于 decoder-only Transformer，忽略 allocator 开销：

`KV bytes ≈ 2 × L × T × H_kv × D_head × bytes_per_element × batch_size`

系数 2 对应 K 和 V。GQA/MQA 会减少 `H_kv`，从而直接降低 cache 显存和 KV 读取流量。

## 5. KV Cache 和 Prefix Cache 有什么区别？

普通 KV Cache 复用的是**单次自回归请求内部**的历史：decode 时不需要重新计算之前
token 的 K/V。Prefix Cache 复用的是**不同请求之间**的 prefill 状态，前提是它们拥有
相同前缀；本项目要求整个 prompt token 序列完全一致。

## 6. 为什么还要缓存 prefill 的最终 logits？

Prefill 完成后，下一个 token 是根据 prompt 最后一个位置的 logits 采样的。如果只缓存
K/V，仍需额外执行模型计算才能重新得到 logits。把它一并缓存后，精确 Prefix 命中可以
完整跳过 prefill。

## 7. 为什么要缓存 `prev_embedding`？

当前 nanochat 有 smear 操作，会把前一个 token 的 embedding 混入当前 token，因此
K/V 并不是完整的推理状态。必须同时恢复 `prev_embedding`，否则 warm Prefix 请求可能
与 cold 请求产生不同结果。

## 8. 为什么只实现精确前缀匹配？

这样最容易保证正确性，也便于审查实现。生产级引擎可能使用 block-level hashing 或
radix tree 做最长前缀匹配，但它们会带来更复杂的显存管理、引用计数和淘汰逻辑，超出
这个小型学习项目的目标。

## 9. 为什么 Prefix Cache 主要改善 TTFT，而不是 TPOT？

因为它跳过的是 prefill。一旦进入 decode，cached 和 uncached 路径执行的是同一个
单 token decode loop，所以稳定阶段的单 token 延迟应接近。2048-token 实测中，warm
命中将 TTFT 从 48.06 ms 降到 1.35 ms，而 TPOT 为 6.420 ms，baseline 为
6.416 ms。

## 10. 为什么 CUDA 计时必须同步？

CUDA kernel 是异步启动的。如果只用 CPU wall-clock 计时而不同步，测到的大多只是
kernel 入队时间。在计时区间前后执行同步，才能让 wall-clock 覆盖 GPU 的真实执行时间。

## 11. 为什么 benchmark 前要 warmup？

第一次运行可能包含 kernel 加载、allocator 活动、cache 初始化和 autotuning。预热可以
减少这些一次性开销，使后续重复测量更能代表稳态性能。

## 12. 为什么取中位数？

延迟测量会受到操作系统和运行时抖动影响。样本数量较小时，中位数比简单平均值更不容易
被异常值带偏。

## 13. 这个实现有哪些限制？

Prefix Cache 只支持精确匹配、常驻 GPU、仅属于单个 Engine 实例，并且 LRU 以条目数
而不是字节预算限制容量。它是学习型实现，不是生产级 serving cache。

另外，单 cache 路径在本次工作负载下没有降低实测 CUDA 峰值已分配显存：2048-token
场景中，baseline 和快速路径分别为 2020.3 MB 与 2023.3 MB。即使省去了 KV 复制，
当 prefill activation 决定显存高水位时，整体峰值仍可能不下降。

## 14. 下一步会做什么？

合理的扩展包括按字节预算淘汰、block-level 最长前缀匹配或连续批处理。我只会在当前
benchmark 明确显示瓶颈且正确性测试持续通过后继续扩展。根据本次结果，按字节预算的
淘汰策略是最直接的下一步，因为 warm Prefix 收益很大，而缓存的 KV 状态常驻 GPU。
