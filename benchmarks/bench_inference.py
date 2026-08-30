"""Benchmark nanochat Engine vs the lightweight optimized engine.

Example:
    python benchmarks/bench_inference.py --prompt-lengths 128 512 1024 \
        --max-new-tokens 64 --repeats 5 --output results/bench.csv

Run this inside an environment where ``nanochat`` is installed and a base
checkpoint is available to nanochat's checkpoint manager.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import time

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

from nanochat_optim.engine import OptimizedEngine


BASE_TEXT = (
    "Large language model inference alternates between a compute-heavy prefill "
    "phase and an autoregressive decode phase. KV cache avoids recomputing "
    "historical keys and values. "
)


@dataclass
class Measurement:
    engine: str
    workload: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    output_tok_s: float
    peak_memory_mb: float


def sync(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def build_prompt(tokenizer, target_tokens: int, bos_token_id: int) -> list[int]:
    if target_tokens < 2:
        raise ValueError("prompt length must be >= 2")
    text = BASE_TEXT
    tokens = tokenizer.encode(text, prepend=bos_token_id)
    while len(tokens) < target_tokens:
        text += BASE_TEXT
        tokens = tokenizer.encode(text, prepend=bos_token_id)
    return tokens[:target_tokens]


def consume_generation(engine, prompt, max_new_tokens, device_type) -> Measurement:
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    stream = engine.generate(
        prompt,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    sync(device_type)
    t0 = time.perf_counter()
    generated = 0

    try:
        next(stream)
        generated = 1
        sync(device_type)
        t_first = time.perf_counter()
    except StopIteration:
        t_first = time.perf_counter()
        sync(device_type)
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if device_type == "cuda" else 0.0
        )
        return Measurement("", "", len(prompt), 0, (t_first-t0)*1e3, 0.0,
                           (t_first-t0)*1e3, 0.0, peak_mb)

    for _ in stream:
        generated += 1

    sync(device_type)
    t1 = time.perf_counter()

    ttft = t_first - t0
    e2e = t1 - t0
    tpot = (e2e - ttft) / max(generated - 1, 1)
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if device_type == "cuda" else 0.0
    )
    return Measurement(
        engine="",
        workload="",
        prompt_tokens=len(prompt),
        generated_tokens=generated,
        ttft_ms=ttft * 1e3,
        tpot_ms=tpot * 1e3,
        e2e_ms=e2e * 1e3,
        output_tok_s=(generated / e2e) if e2e > 0 else 0.0,
        peak_memory_mb=peak_mb,
    )


def median_measurements(rows: list[Measurement], engine: str, workload: str) -> Measurement:
    first = rows[0]
    return Measurement(
        engine=engine,
        workload=workload,
        prompt_tokens=first.prompt_tokens,
        generated_tokens=int(statistics.median(r.generated_tokens for r in rows)),
        ttft_ms=statistics.median(r.ttft_ms for r in rows),
        tpot_ms=statistics.median(r.tpot_ms for r in rows),
        e2e_ms=statistics.median(r.e2e_ms for r in rows),
        output_tok_s=statistics.median(r.output_tok_s for r in rows),
        peak_memory_mb=statistics.median(r.peak_memory_mb for r in rows),
    )


def run_repeated(engine, prompt, args, device_type, label, workload):
    # One untimed warmup to reduce first-run kernel/autotune noise.
    list(engine.generate(prompt, num_samples=1, max_tokens=min(8, args.max_new_tokens), temperature=0.0))
    sync(device_type)

    rows = [
        consume_generation(engine, prompt, args.max_new_tokens, device_type)
        for _ in range(args.repeats)
    ]
    return median_measurements(rows, label, workload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefix-cache-entries", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/bench.csv"))
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model("base", device, phase="eval")
    bos = tokenizer.get_bos_token_id()

    baseline = Engine(model, tokenizer)
    fast_path = OptimizedEngine(
        model,
        tokenizer,
        enable_single_cache_fast_path=True,
        prefix_cache_entries=0,
    )
    prefix_engine = OptimizedEngine(
        model,
        tokenizer,
        enable_single_cache_fast_path=True,
        prefix_cache_entries=args.prefix_cache_entries,
    )

    output_rows: list[Measurement] = []

    for prompt_len in args.prompt_lengths:
        prompt = build_prompt(tokenizer, prompt_len, bos)

        output_rows.append(
            run_repeated(baseline, prompt, args, device_type, "baseline", "normal")
        )
        output_rows.append(
            run_repeated(fast_path, prompt, args, device_type, "single-cache-fast-path", "normal")
        )

        # Populate exact prefix cache once, then benchmark cache hits. Clear first
        # so every prompt length has an explicit cold->warm transition.
        prefix_engine.clear_prefix_cache()
        list(prefix_engine.generate(prompt, num_samples=1, max_tokens=1, temperature=0.0))
        sync(device_type)
        output_rows.append(
            run_repeated(prefix_engine, prompt, args, device_type, "prefix-cache", "warm-prefix")
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(output_rows[0]).keys()))
        writer.writeheader()
        for row in output_rows:
            writer.writerow(asdict(row))

    print(f"{'engine':24} {'workload':12} {'prompt':>7} {'TTFT(ms)':>10} "
          f"{'TPOT(ms)':>10} {'tok/s':>10} {'peak(MB)':>10}")
    for row in output_rows:
        print(
            f"{row.engine:24} {row.workload:12} {row.prompt_tokens:7d} "
            f"{row.ttft_ms:10.2f} {row.tpot_ms:10.2f} "
            f"{row.output_tok_s:10.2f} {row.peak_memory_mb:10.1f}"
        )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
