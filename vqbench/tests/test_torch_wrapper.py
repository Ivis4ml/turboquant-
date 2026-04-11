"""
PyTorch wrapper and HuggingFace integration tests — Phase 8.

Tests:
  - QuantizedKVCache update/get roundtrip
  - VQBenchCache satisfies transformers.Cache protocol
  - Multi-layer, multi-head support
  - Compression ratio > 1
  - make_vqbench_cache auto-detect from config
"""

import numpy as np
import pytest
import torch

from vqbench.torch_wrapper.module import QuantizedKVCache, _get_method_class
from vqbench.torch_wrapper.hook import VQBenchCache, make_vqbench_cache

HEAD_DIM = 128
NUM_KV_HEADS = 4
NUM_LAYERS = 2
SEQ_LEN = 32


class TestQuantizedKVCache:
    def test_update_and_get(self):
        """Basic update → get roundtrip produces correct shapes."""
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=NUM_LAYERS, num_bits_key=2, num_bits_value=2,
        )
        keys = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        values = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)

        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        assert v_out.shape == (1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)

    def test_incremental_update(self):
        """Sequential updates accumulate tokens."""
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1, num_bits_key=3, num_bits_value=3,
        )
        # First chunk
        k1 = torch.randn(1, NUM_KV_HEADS, 10, HEAD_DIM)
        v1 = torch.randn(1, NUM_KV_HEADS, 10, HEAD_DIM)
        cache.update(k1, v1, layer_idx=0)
        assert cache.seq_len(0) == 10

        # Second chunk
        k2 = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM)
        v2 = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM)
        k_out, v_out = cache.update(k2, v2, layer_idx=0)
        assert cache.seq_len(0) == 15
        assert k_out.shape == (1, NUM_KV_HEADS, 15, HEAD_DIM)

    def test_multi_layer(self):
        """Each layer has independent cache."""
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=NUM_LAYERS, num_bits_key=2, num_bits_value=2,
        )
        k = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)

        cache.update(k, v, layer_idx=0)
        cache.update(k, v, layer_idx=1)

        assert cache.seq_len(0) == SEQ_LEN
        assert cache.seq_len(1) == SEQ_LEN

    def test_compression_ratio(self):
        """Compression ratio should be > 1."""
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1, num_bits_key=2, num_bits_value=2,
        )
        k = torch.randn(1, NUM_KV_HEADS, 100, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 100, HEAD_DIM)
        cache.update(k, v, layer_idx=0)

        ratio = cache.compression_ratio()
        assert ratio > 1.0, f"Compression ratio = {ratio:.2f}"

    def test_clear(self):
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1, num_bits_key=2, num_bits_value=2,
        )
        k = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        cache.update(k, v, layer_idx=0)
        assert cache.seq_len(0) == SEQ_LEN

        cache.clear()
        assert cache.seq_len(0) == 0

    def test_dtype_preservation(self):
        """Output dtype matches input dtype."""
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1, num_bits_key=2, num_bits_value=2,
        )
        k = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
        v = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.dtype == torch.bfloat16
        assert v_out.dtype == torch.bfloat16

    def test_update_rejects_batch_gt_one(self):
        """batch_size != 1 must raise instead of silently corrupting batches > 0."""
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1, num_bits_key=2, num_bits_value=2,
        )
        k = torch.randn(2, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        with pytest.raises(ValueError, match="batch_size == 1"):
            cache.update(k, v, layer_idx=0)

    def test_get_rejects_batch_gt_one(self):
        cache = QuantizedKVCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1, num_bits_key=2, num_bits_value=2,
        )
        with pytest.raises(ValueError, match="batch_size == 1"):
            cache.get(layer_idx=0, batch_size=2)

    def test_block_b64_registered(self):
        """BlockTurboQuantMSE-B64 is the headline Qwen3.5-4B config — registry must expose it."""
        cls = _get_method_class("BlockTurboQuantMSE-B64")
        q = cls(d=256, num_bits=4, seed=0)
        assert q.block_size == 64

        cache = QuantizedKVCache(
            head_dim=256, num_kv_heads=2,
            num_layers=1,
            method_key="BlockTurboQuantMSE-B64",
            method_value="TurboQuantMSE",
            num_bits_key=4, num_bits_value=4,
        )
        k = torch.randn(1, 2, 16, 256)
        v = torch.randn(1, 2, 16, 256)
        k_out, _ = cache.update(k, v, layer_idx=0)
        assert k_out.shape == (1, 2, 16, 256)


class TestVQBenchCache:
    def test_is_cache_subclass(self):
        """VQBenchCache must be a transformers.Cache subclass."""
        from transformers import Cache
        cache = VQBenchCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=NUM_LAYERS,
        )
        assert isinstance(cache, Cache)

    def test_layer_update_protocol(self):
        """Layer update() returns (keys, values) tuple."""
        cache = VQBenchCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1,
        )
        k = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        layer = cache.layers[0]
        result = layer.update(k, v)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_seq_length(self):
        cache = VQBenchCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1,
        )
        assert cache.get_seq_length() == 0
        k = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        cache.layers[0].update(k, v)
        assert cache.get_seq_length() == SEQ_LEN

    def test_reset(self):
        cache = VQBenchCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1,
        )
        k = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        cache.layers[0].update(k, v)
        cache.reset()
        assert cache.get_seq_length() == 0

    def test_layer_update_rejects_batch_gt_one(self):
        cache = VQBenchCache(
            head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS,
            num_layers=1,
        )
        k = torch.randn(2, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(2, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        with pytest.raises(ValueError, match="batch_size == 1"):
            cache.layers[0].update(k, v)


class TestMakeVQBenchCache:
    def test_from_mock_config(self):
        """make_vqbench_cache extracts params from model config."""

        class MockConfig:
            num_hidden_layers = 4
            hidden_size = 512
            num_attention_heads = 4
            num_key_value_heads = 2
            head_dim = 128

        cache = make_vqbench_cache(MockConfig())
        assert isinstance(cache, VQBenchCache)

    def test_auto_head_dim(self):
        """Compute head_dim from hidden_size / num_heads when not explicit."""

        class MockConfig:
            num_hidden_layers = 2
            hidden_size = 1024
            num_attention_heads = 8
            num_key_value_heads = 4

        cache = make_vqbench_cache(MockConfig())
        assert cache._head_dim == 128  # 1024 / 8
