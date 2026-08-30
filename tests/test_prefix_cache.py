import torch

from nanochat_optim.prefix_cache import ExactPrefixCache


class FakeKVCache:
    def __init__(self, batch_size=1, max_seq_len=8, prompt_len=0):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = 2
        self.k_cache = torch.zeros(2, batch_size, max_seq_len, 2, 4)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.cache_seqlens = torch.full((batch_size,), prompt_len, dtype=torch.int32)
        self.prev_embedding = None

    def get_pos(self):
        return int(self.cache_seqlens[0].item())


def make_source(tokens):
    source = FakeKVCache(batch_size=1, max_seq_len=len(tokens), prompt_len=len(tokens))
    source.k_cache.uniform_()
    source.v_cache.uniform_()
    source.prev_embedding = torch.randn(1, 1, 8)
    logits = torch.randn(1, 32)
    return source, logits


def test_exact_hit_restores_state_for_larger_batch():
    tokens = [1, 2, 3, 4]
    source, logits = make_source(tokens)
    cache = ExactPrefixCache(capacity=2)
    cache.put(tokens, source, logits)

    target = FakeKVCache(batch_size=3, max_seq_len=8, prompt_len=0)
    restored_logits = cache.restore(tokens, target)

    assert restored_logits is not None
    assert restored_logits.shape == (3, 32)
    assert target.get_pos() == len(tokens)
    for b in range(3):
        assert torch.equal(target.k_cache[:, b, : len(tokens)], source.k_cache[:, 0])
        assert torch.equal(target.v_cache[:, b, : len(tokens)], source.v_cache[:, 0])
        assert torch.equal(target.prev_embedding[b], source.prev_embedding[0])
    assert cache.hits == 1


def test_miss_does_not_modify_target():
    source, logits = make_source([1, 2, 3])
    cache = ExactPrefixCache(capacity=2)
    cache.put([1, 2, 3], source, logits)
    target = FakeKVCache(batch_size=1, max_seq_len=8, prompt_len=0)

    before = target.k_cache.clone()
    assert cache.restore([1, 2, 9], target) is None
    assert torch.equal(target.k_cache, before)
    assert cache.misses == 1


def test_lru_eviction():
    cache = ExactPrefixCache(capacity=2)
    for tokens in ([1], [2], [3]):
        source, logits = make_source(list(tokens))
        cache.put(list(tokens), source, logits)

    assert len(cache) == 2
    target = FakeKVCache(batch_size=1, max_seq_len=4, prompt_len=0)
    assert cache.restore([1], target) is None
