"""Lightweight inference optimizations for nanochat."""

from .prefix_cache import ExactPrefixCache, PrefixCacheEntry

__all__ = ["ExactPrefixCache", "PrefixCacheEntry"]
