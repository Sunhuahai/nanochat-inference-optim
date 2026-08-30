# nanochat-inference-optim

为 [karpathy/nanochat](https://github.com/karpathy/nanochat) 实现的一组小型推理优化，
重点是便于理解、验证和面试讲解。

本项目的目标**不是**重新实现 vLLM，而是聚焦两个可以端到端分析和测试的改动。

实现基于上游 nanochat `master` 的提交
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`（核对日期：2026-08-30）。

## 优化内容

### 1. 单样本 KV Cache 快速路径

当前 nanochat 的 `Engine.generate()` 会先以 batch size 1 将 prompt prefill 到临时
KV Cache，再分配 decode cache，并把 prompt KV 复制过去。当同一个 prompt 需要扩展成
多个样本时，这种设计很有用；但在 `num_samples=1` 时，这次复制是不必要的。

本项目改为：

```text
原始实现，num_samples=1
prompt -> 临时 prefill KV -> 复制 -> 最终 decode KV -> decode

优化实现
prompt ---------------------> 最终 decode KV -> decode
```

这样可以省去一个 prompt 大小的 KV 分配，以及一次与 prompt 长度成正比的 KV 复制。
预期主要影响 **TTFT** 和瞬时显存，对 decode 阶段的 TPOT 影响很小。

### 2. 精确匹配 Prefix KV Cache

当完全相同的 token 序列再次出现时，优化后的引擎可以恢复已有 prefill 状态，
无需重新计算。

```text
prompt tokens
    |
    v
BLAKE2 哈希 -> LRU 查询 ---- 命中 ----> KV + logits + smear 状态
                    |
                   未命中
                    v
                 prefill
```

Cache 保存：

- K cache
- V cache
- prefill 最后一个位置的 logits
- nanochat smear 操作使用的 `prev_embedding`

该实现刻意只支持**完全匹配**，不引入 radix tree、分页、自定义 CUDA 或连续批处理。

## 为什么适合面试讲解

代码量足够小，可以逐行说明，同时自然关联以下主题：

- KV Cache 显存计算
- GQA 与 KV 显存优化
- Prefill 与 Decode
- TTFT 与 TPOT
- GPU 异步计时
- Prefix Cache 正确性
- Cache 显存与延迟的权衡

常见追问及回答要点见
[`docs/interview_questions.md`](docs/interview_questions.md)。

## 仓库结构

```text
nanochat_optim/
  engine.py           # 优化后的 Engine
  prefix_cache.py     # 精确 prompt LRU cache
benchmarks/
  bench_inference.py  # TTFT / TPOT / 吞吐 / 峰值显存 benchmark
  check_correctness.py # greedy cold/warm 等价性检查
tests/
  test_prefix_cache.py
docs/
  experiment_plan.md
  interview_questions.md
```

## 环境安装

依赖通过 `uv` 锁定并安装在仓库内的 `.venv` 中。安装脚本还会把经过测试的
nanochat 上游提交检出到项目内的 `.upstream/`；不会向系统 Python 安装任何包。

```bash
bash scripts/setup_env.sh
```

`uv` 会在不同项目之间复用全局下载缓存，但每个项目拥有独立的安装环境和上游源码。
运行命令时由 `scripts/run_env.sh` 同时选择二者。模型文件沿用 nanochat 的标准目录
`~/.cache/nanochat`，例如 base checkpoint 路径为：
`~/.cache/nanochat/base_checkpoints/d14/model_002192.pt`。

## 运行单元测试

Prefix Cache 的独立测试不需要 nanochat checkpoint：

```bash
scripts/run_env.sh pytest -q
```

## 与上游实现进行正确性检查

准备好 nanochat base checkpoint 后运行：

```bash
scripts/run_env.sh python benchmarks/check_correctness.py
```

该脚本使用 greedy decoding，要求优化后的 cold path 和 warm Prefix Cache path
生成的 token 都与上游实现完全一致。

## 运行 benchmark

准备好 nanochat base checkpoint 后运行：

```bash
scripts/run_env.sh python benchmarks/bench_inference.py \
  --prompt-lengths 128 512 1024 2048 \
  --max-new-tokens 64 \
  --repeats 5 \
  --output results/bench.csv
```

Benchmark 对比：

1. 上游 `Engine`；
2. 关闭 Prefix Cache 的单 cache 快速路径；
3. 精确 Prefix Cache 的 warm hit。

报告以下指标的中位数：

- TTFT（首 token 延迟）
- TPOT（首 token 之后的平均单 token 延迟）
- E2E latency（端到端延迟）
- output tokens/s（输出吞吐）
- peak CUDA memory allocated（CUDA 峰值已分配显存）

## 实测性能

在目标 GPU 实测前，不应预先编造或写死性能提升数字。

预期趋势：

- 单 cache 快速路径：省去的 KV 复制工作量随 prompt 变长而增加，但延迟测量仍会有
  噪声；decode TPOT 基本不变；
- 精确 Prefix Cache：对重复长 prompt 可显著降低 TTFT，代价是将 KV 状态保留在
  GPU 显存中。

以下数据在 NVIDIA GeForce RTX 4060 8GB、PyTorch 2.9.1+cu128、nanochat d14、
greedy decoding、生成 64 个 token 的条件下测得，每项重复 5 次并取中位数：

| Prompt | Baseline TTFT | 快速路径 TTFT | Prefix 命中 TTFT | 峰值显存（baseline / fast / prefix） |
|---:|---:|---:|---:|---:|
| 128 | 8.51 ms | 7.99 ms（-6.15%） | 0.18 ms（-97.90%） | 1168.9 / 1169.1 / 1160.2 MB |
| 512 | 15.23 ms | 14.98 ms（-1.61%） | 0.41 ms（-97.30%） | 1315.7 / 1319.2 / 1196.5 MB |
| 1024 | 25.75 ms | 25.26 ms（-1.91%） | 0.72 ms（-97.19%） | 1554.0 / 1557.1 / 1245.0 MB |
| 2048 | 48.06 ms | 46.72 ms（-2.79%） | 1.35 ms（-97.19%） | 2020.3 / 2023.3 / 1343.0 MB |

Decode TPOT 保持在约 6.1-6.4 ms。单 cache 快速路径降低了 TTFT，但在该环境中
没有降低峰值显存高水位；Prefix Cache 数据专指 token 序列完全相同的重复 prompt。

## 简历表述

> 分析 nanochat 自回归推理链路并搭建 TTFT、TPOT、吞吐和显存 benchmark；
> 在 RTX 4060、nanochat d14 上，通过移除单样本场景中冗余的 KV 复制，将
> 2048-token prompt 的 TTFT 从 48.06 ms 降至 46.72 ms（2.8%），并通过精确
> Prefix Cache 将重复 prompt 的 TTFT 降至 1.35 ms（97.2%）；验证上游、优化
> cold path 和 warm-cache path 的 greedy token 完全一致。

## 项目边界

本项目不是生产级 serving engine，以下内容不在范围内：

- PagedAttention
- 连续批处理
- 分布式推理
- 自定义 CUDA kernel
- CPU KV swapping
- speculative decoding

这些都是合理的后续方向，但会使这个学习型项目明显膨胀。

## 第三方归属

本项目是 nanochat 的独立教育性扩展，详见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
