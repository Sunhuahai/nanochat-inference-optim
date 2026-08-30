"""Greedy correctness check against upstream nanochat Engine.

This requires a nanochat base checkpoint. It compares:
1. upstream Engine;
2. optimized cold path;
3. optimized warm exact-prefix path.
"""

from __future__ import annotations

import argparse

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

from nanochat_optim.engine import OptimizedEngine


def collect(engine, tokens, max_tokens):
    out = []
    for token_column, _ in engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=0.0,
    ):
        out.append(token_column[0])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="Explain KV cache in one short paragraph.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model("base", device, phase="eval")
    tokens = tokenizer.encode(args.prompt, prepend=tokenizer.get_bos_token_id())

    baseline = Engine(model, tokenizer)
    optimized = OptimizedEngine(model, tokenizer, prefix_cache_entries=2)

    ref = collect(baseline, tokens, args.max_new_tokens)
    cold = collect(optimized, tokens, args.max_new_tokens)
    warm = collect(optimized, tokens, args.max_new_tokens)

    print(f"baseline vs optimized cold: {ref == cold}")
    print(f"baseline vs optimized warm: {ref == warm}")
    print(f"prefix cache hits: {optimized.prefix_cache.hits}")

    if ref != cold or ref != warm:
        raise SystemExit("correctness check failed")


if __name__ == "__main__":
    main()
