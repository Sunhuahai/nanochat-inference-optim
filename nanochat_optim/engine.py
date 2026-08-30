"""A deliberately small optimized inference Engine for nanochat.

Two changes relative to nanochat's reference ``Engine``:

1. Single-sample KV fast path
   For ``num_samples == 1`` the reference engine first prefills into a
   prompt-sized KV cache, allocates a second decode cache, copies all prompt KV,
   and then frees the first cache. Here prefill writes directly into the final
   decode cache, avoiding one allocation and one full-prefix KV copy.

2. Exact Prefix KV Cache
   Repeated token-identical prompts can reuse prefill KV state and the last
   prefill logits. The cache also preserves ``prev_embedding``, which is part of
   nanochat's inference state because of the smear operation.

The generation state machine is intentionally kept close to nanochat's Engine
so that the optimization remains easy to audit and explain in an interview.

Portions of the generation loop are adapted from karpathy/nanochat (MIT).
See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import Engine, KVCache, RowState, sample_next_token, use_calculator

from .prefix_cache import ExactPrefixCache


class OptimizedEngine(Engine):
    """nanochat Engine with two small, independently benchmarkable optimizations."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        enable_single_cache_fast_path: bool = True,
        prefix_cache_entries: int = 4,
    ):
        super().__init__(model, tokenizer)
        self.enable_single_cache_fast_path = enable_single_cache_fast_path
        self.prefix_cache = ExactPrefixCache(prefix_cache_entries)

    def clear_prefix_cache(self) -> None:
        self.prefix_cache.clear()

    def _new_kv_cache(self, batch_size: int, seq_len: int) -> KVCache:
        m = self.model.config
        return KVCache(
            batch_size=batch_size,
            num_heads=m.n_kv_head,
            seq_len=seq_len,
            head_dim=m.n_embd // m.n_head,
            num_layers=m.n_layer,
            device=self.model.get_device(),
            dtype=COMPUTE_DTYPE,
        )

    @torch.inference_mode()
    def _prepare_prefill(
        self,
        tokens: list[int],
        num_samples: int,
        max_tokens: int | None,
    ) -> tuple[torch.Tensor, KVCache, bool]:
        """Return ``(logits, decode_cache, prefix_hit)`` after prompt prefill."""
        device = self.model.get_device()
        kv_length_hint = (
            len(tokens) + max_tokens
            if max_tokens is not None
            else self.model.config.sequence_len
        )

        # Allocate the final decode cache first. This also gives prefix-cache hits
        # a destination without needing any temporary prefill cache.
        decode_cache = self._new_kv_cache(num_samples, kv_length_hint)

        # Optimization 2: exact cross-request prefix reuse.
        cached_logits = self.prefix_cache.restore(tokens, decode_cache)
        if cached_logits is not None:
            return cached_logits, decode_cache, True

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Optimization 1: batch=1 can prefill directly into the final cache.
        if num_samples == 1 and self.enable_single_cache_fast_path:
            logits = self.model.forward(ids, kv_cache=decode_cache)[:, -1, :]
            self.prefix_cache.put(tokens, decode_cache, logits)
            return logits, decode_cache, False

        # Reference-style path for num_samples > 1 (or when the fast path is
        # disabled): prefill batch=1, then replicate the KV state.
        prefill_cache = self._new_kv_cache(1, len(tokens))
        logits = self.model.forward(ids, kv_cache=prefill_cache)[:, -1, :]
        self.prefix_cache.put(tokens, prefill_cache, logits)
        decode_cache.prefill(prefill_cache)
        del prefill_cache
        return logits.expand(num_samples, -1), decode_cache, False

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42,
    ):
        """Generate tokens with the same public contract as nanochat.Engine."""
        assert isinstance(tokens, list) and tokens and isinstance(tokens[0], int), (
            "expecting non-empty list of ints"
        )
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Special tokens used by nanochat's calculator/tool state machine.
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        logits, kv_cache_decode, _ = self._prepare_prefill(
            tokens, num_samples, max_tokens
        )
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = (
                    state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                )
                token_column.append(next_token)
                state.current_tokens.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
