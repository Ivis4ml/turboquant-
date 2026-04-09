"""
HuggingFace transformers integration: VQBenchCache subclassing Cache.

Phase 8.2: Drop-in replacement for DynamicCache with VQ compression.
Compatible with transformers >= 5.5.0 (layer-based Cache API).

Usage:
    from transformers import AutoModelForCausalLM
    from vqbench.torch_wrapper.hook import apply_quantized_cache

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B")
    cache = apply_quantized_cache(model, method_key="TurboQuantProd", num_bits=3)
    output = model.generate(input_ids, max_new_tokens=1000, past_key_values=cache)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers import Cache
from transformers.cache_utils import CacheLayerMixin

from vqbench.core.base import VectorQuantizer
from vqbench.kv_cache.compressor import KVCacheCompressor
from vqbench.torch_wrapper.module import QuantizedKVCache, _get_method_class


class VQBenchCacheLayer(CacheLayerMixin):
    """
    Single-layer cache backed by VQ compression.

    Implements transformers CacheLayerMixin protocol.
    Each layer holds per-head KVCacheCompressors.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_cls: type[VectorQuantizer],
        val_cls: type[VectorQuantizer],
        num_bits_key: int,
        num_bits_value: int,
        layer_seed: int,
    ) -> None:
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self._compressors: list[KVCacheCompressor] = []
        for head in range(num_kv_heads):
            head_seed = layer_seed + head
            k_q = key_cls(d=head_dim, num_bits=num_bits_key, seed=head_seed)
            v_q = val_cls(d=head_dim, num_bits=num_bits_value, seed=head_seed)
            self._compressors.append(KVCacheCompressor(k_q, v_q))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress new KV, return decompressed full cache."""
        device = key_states.device
        dtype = key_states.dtype

        k_np = key_states[0].detach().cpu().to(torch.float64).numpy()
        v_np = value_states[0].detach().cpu().to(torch.float64).numpy()

        for h in range(self.num_kv_heads):
            self._compressors[h].compress(k_np[h], v_np[h])

        return self._get_all(device, dtype, batch_size=key_states.shape[0])

    def _get_all(
        self, device: torch.device, dtype: torch.dtype, batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_keys = []
        all_values = []
        for h in range(self.num_kv_heads):
            all_keys.append(self._compressors[h].get_keys())
            all_values.append(self._compressors[h].get_values())

        seq_len = all_keys[0].shape[0] if len(all_keys[0]) > 0 else 0
        if seq_len == 0:
            shape = (batch_size, self.num_kv_heads, 0, self.head_dim)
            return (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )

        k_np = np.stack(all_keys)
        v_np = np.stack(all_values)
        k_t = torch.from_numpy(k_np).unsqueeze(0).expand(batch_size, -1, -1, -1)
        v_t = torch.from_numpy(v_np).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return k_t.to(dtype=dtype, device=device), v_t.to(dtype=dtype, device=device)

    def get_seq_length(self) -> int:
        return self._compressors[0].num_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def reset(self) -> None:
        for comp in self._compressors:
            comp.clear()

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """No lazy init needed — compressors are pre-built."""
        pass

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return (cache_length, sliding_window_length) for attention mask."""
        seq_len = self.get_seq_length()
        return seq_len, seq_len


class VQBenchCache(Cache):
    """
    Transformers-compatible Cache backed by VQ compression.

    Uses the transformers 5.5.0+ layer-based Cache API.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        num_layers: int,
        method_key: str = "TurboQuantProd",
        method_value: str = "TurboQuantMSE",
        num_bits_key: int = 3,
        num_bits_value: int = 3,
        seed: int = 42,
    ) -> None:
        key_cls = _get_method_class(method_key)
        val_cls = _get_method_class(method_value)

        layers = []
        for layer_idx in range(num_layers):
            layer_seed = seed + layer_idx * 1000
            layers.append(VQBenchCacheLayer(
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                key_cls=key_cls,
                val_cls=val_cls,
                num_bits_key=num_bits_key,
                num_bits_value=num_bits_value,
                layer_seed=layer_seed,
            ))

        super().__init__(layers=layers)
        self._head_dim = head_dim
        self._num_kv_heads = num_kv_heads
        self._num_layers = num_layers

    def memory_bytes(self) -> int:
        total_bits = 0
        for layer in self.layers:
            for comp in layer._compressors:
                total_bits += comp.storage_bits()
        return total_bits // 8

    def compression_ratio(self) -> float:
        total_tokens = sum(
            comp.num_tokens
            for layer in self.layers
            for comp in layer._compressors
        )
        if total_tokens == 0:
            return 0.0
        fp16_bits = total_tokens * self._head_dim * 16 * 2
        compressed_bits = sum(
            comp.storage_bits()
            for layer in self.layers
            for comp in layer._compressors
        )
        return fp16_bits / max(compressed_bits, 1)


def make_vqbench_cache(
    model_config,
    method_key: str = "TurboQuantProd",
    method_value: str = "TurboQuantMSE",
    num_bits_key: int = 3,
    num_bits_value: int = 3,
    seed: int = 42,
) -> VQBenchCache:
    """Create a VQBenchCache from a HuggingFace model config."""
    num_layers = getattr(model_config, "num_hidden_layers", 32)
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_attention_heads
        head_dim = hidden_size // num_heads

    num_kv_heads = getattr(
        model_config, "num_key_value_heads",
        getattr(model_config, "num_attention_heads", 8),
    )

    return VQBenchCache(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        method_key=method_key,
        method_value=method_value,
        num_bits_key=num_bits_key,
        num_bits_value=num_bits_value,
        seed=seed,
    )


def apply_quantized_cache(
    model,
    method_key: str = "TurboQuantProd",
    method_value: str = "TurboQuantMSE",
    num_bits: int = 3,
    num_bits_key: int | None = None,
    num_bits_value: int | None = None,
    seed: int = 42,
) -> VQBenchCache:
    """
    Create a VQBenchCache configured for a HuggingFace model.

    Returns the cache — pass to model.generate(past_key_values=cache).
    """
    return make_vqbench_cache(
        model.config,
        method_key=method_key,
        method_value=method_value,
        num_bits_key=num_bits_key or num_bits,
        num_bits_value=num_bits_value or num_bits,
        seed=seed,
    )
