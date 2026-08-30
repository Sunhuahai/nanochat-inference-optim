"""Exact prompt KV cache for nanochat inference.

This module intentionally keeps the design small:
- cache only *exactly identical* tokenized prompts;
- keep a bounded LRU on the same device as the model;
- cache the final prefill logits as well as KV state;
- preserve nanochat's ``prev_embedding`` smear state.

It is not intended to compete with vLLM's prefix caching. The point of the
implementation is to make cross-request KV reuse concrete and benchmarkable.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import torch


def _prompt_key(tokens: Iterable[int]) -> str:
    """Return a stable hash for a token-id sequence.

    Token IDs are encoded as unsigned 32-bit little-endian integers. nanochat's
    vocabulary is far smaller than 2**32, so this is simple and unambiguous.
    """
    h = hashlib.blake2b(digest_size=16)
    for token in tokens:
        value = int(token)
        if value < 0 or value >= 2**32:
            raise ValueError(f"token id out of uint32 range: {value}")
        h.update(value.to_bytes(4, byteorder="little", signed=False))
    return h.hexdigest()


@dataclass
class PrefixCacheEntry:
    """Cached inference state after prefill of one prompt."""

    prompt_tokens: tuple[int, ...]
    prompt_len: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    last_logits: torch.Tensor
    prev_embedding: torch.Tensor | None

    @property
    def nbytes(self) -> int:
        tensors = [self.k_cache, self.v_cache, self.last_logits]
        if self.prev_embedding is not None:
            tensors.append(self.prev_embedding)
        return sum(t.numel() * t.element_size() for t in tensors)


class ExactPrefixCache:
    """A tiny LRU cache for exact prompt reuse.

    Parameters
    ----------
    capacity:
        Maximum number of prompts retained. ``0`` disables caching.
    """

    def __init__(self, capacity: int = 4):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def put(self, tokens: list[int], kv_cache: Any, last_logits: torch.Tensor) -> None:
        """Snapshot the prefill state from a batch-size-1 nanochat KV cache."""
        if self.capacity == 0:
            return
        if kv_cache.batch_size != 1:
            raise ValueError("prefix entries must be captured from batch_size=1 prefill")

        prompt_len = int(kv_cache.get_pos())
        if prompt_len != len(tokens):
            raise ValueError(
                f"KV position ({prompt_len}) does not match prompt length ({len(tokens)})"
            )

        key = _prompt_key(tokens)
        entry = PrefixCacheEntry(
            prompt_tokens=tuple(tokens),
            prompt_len=prompt_len,
            k_cache=kv_cache.k_cache[:, :1, :prompt_len].detach().clone(),
            v_cache=kv_cache.v_cache[:, :1, :prompt_len].detach().clone(),
            last_logits=last_logits[:1].detach().clone(),
            prev_embedding=(
                kv_cache.prev_embedding.detach().clone()
                if kv_cache.prev_embedding is not None
                else None
            ),
        )

        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def restore(self, tokens: list[int], target_kv_cache: Any) -> torch.Tensor | None:
        """Restore an exact prompt into ``target_kv_cache``.

        Returns cached last-token logits on a hit; returns ``None`` on a miss.
        The target cache may have any batch size: the single cached prefix is
        broadcast into all rows, matching nanochat Engine's prompt replication.
        """
        if self.capacity == 0:
            self.misses += 1
            return None

        key = _prompt_key(tokens)
        entry = self._entries.get(key)
        if entry is None or entry.prompt_tokens != tuple(tokens):
            self.misses += 1
            return None

        if target_kv_cache.get_pos() != 0:
            raise ValueError("target KV cache must be empty before prefix restore")
        if target_kv_cache.max_seq_len < entry.prompt_len:
            raise ValueError("target KV cache is shorter than cached prompt")
        if target_kv_cache.n_layers != entry.k_cache.shape[0]:
            raise ValueError("target KV cache layer count does not match cached prefix")

        batch_size = target_kv_cache.batch_size
        target_kv_cache.k_cache[:, :, : entry.prompt_len].copy_(
            entry.k_cache.expand(-1, batch_size, -1, -1, -1)
        )
        target_kv_cache.v_cache[:, :, : entry.prompt_len].copy_(
            entry.v_cache.expand(-1, batch_size, -1, -1, -1)
        )
        target_kv_cache.cache_seqlens.fill_(entry.prompt_len)

        if entry.prev_embedding is None:
            target_kv_cache.prev_embedding = None
        else:
            target_kv_cache.prev_embedding = entry.prev_embedding.expand(
                batch_size, -1, -1
            ).clone()

        self.hits += 1
        self._entries.move_to_end(key)
        return entry.last_logits.expand(batch_size, -1)
