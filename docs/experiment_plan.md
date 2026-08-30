# 实验方案

本项目刻意只验证两个小改动。在目标 GPU 上实测前，不应宣称具体性能提升。

## A. 单样本 KV Cache 快速路径

问题：对 `num_samples=1`，避免临时 prompt cache 和 prompt KV 复制，能否降低
TTFT 与瞬时峰值显存？

对比对象：

- `nanochat.engine.Engine`
- `OptimizedEngine(prefix_cache_entries=0)`

工作负载：

| Prompt tokens | New tokens | Temperature |
|---:|---:|---:|
| 128 | 64 | 0 |
| 512 | 64 | 0 |
| 1024 | 64 | 0 |
| 2048 | 64 | 0 |

主要指标：TTFT、E2E latency、CUDA 峰值已分配显存。

预期行为：省去的 KV 复制量与 prompt 长度成正比，因此长 prompt 下应更容易观察到
收益；但真实延迟仍可能受运行时噪声影响。由于 decode loop 没有变化，decode TPOT
应基本不变。

## B. 精确匹配 Prefix KV Cache

问题：完全相同的长 prompt 被重复使用时，可以节省多少 TTFT？

对每个 prompt 长度执行：

1. 清空 Prefix Cache；
2. 执行一次 cold request 填充 cache；
3. 重复执行相同 request；
4. 将 warm-cache TTFT 与 baseline TTFT 对比。

主要指标：TTFT。次要指标：cache 显存开销与命中率。

预期行为：Prefix Cache 以额外 GPU 显存为代价，省去 prefill 计算；它应主要改善
TTFT，而不是稳定 decode 阶段的 TPOT。

## 正确性检查

使用 greedy decoding（`temperature=0`），比较上游和优化引擎生成的 token 序列。
Prefix Cache 需要分别验证 cold path 与 warm path。

由于 CUDA kernel 是异步启动的，benchmark 脚本会在计时边界执行 CUDA 同步。
