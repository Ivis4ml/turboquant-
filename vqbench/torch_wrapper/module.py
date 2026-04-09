"""
QuantizedKVCache — PyTorch nn.Module for KV-cache compression.

Phase 8.1: Per-head compression using any VQ method.
Supports different methods/bit-widths for keys vs values.

Usage:
    cache = QuantizedKVCache(
        head_dim=128, num_kv_heads=4,
        method_key="TurboQuantProd", method_value="TurboQuantMSE",
        num_bits_key=3, num_bits_value=3,
    )
    cache.update(keys, values, layer_idx=0)
    k, v = cache.get(layer_idx=0)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vqbench.core.base import VectorQuantizer
from vqbench.kv_cache.compressor import KVCacheCompressor


# Method name → class mapping (lazy import to avoid circular deps)
def _get_method_class(name: str) -> type[VectorQuantizer]:
    from vqbench.methods.turboquant.mse import TurboQuantMSE
    from vqbench.methods.turboquant.prod import TurboQuantProd
    from vqbench.methods.turboquant.qjl import QJLQuantizer
    from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
    from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ

    registry = {
        "TurboQuantMSE": TurboQuantMSE,
        "TurboQuantProd": TurboQuantProd,
        "QJL": QJLQuantizer,
        "RaBitQ1Bit": RaBitQ1Bit,
        "ExtRaBitQ": ExtRaBitQ,
    }
    if name not in registry:
        raise ValueError(f"Unknown method '{name}'. Available: {list(registry)}")
    return registry[name]


class QuantizedKVCache(nn.Module):
    """
    Drop-in KV cache with VQ compression, operating per-head.

    Args:
        head_dim: Dimension per attention head.
        num_kv_heads: Number of key/value heads (for GQA).
        method_key: VQ method name for keys (unbiased IP recommended).
        method_value: VQ method name for values (low MSE recommended).
        num_bits_key: Bits per dimension for key compression.
        num_bits_value: Bits per dimension for value compression.
        num_layers: Number of transformer layers.
        seed: Random seed for quantizer initialization.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        method_key: str = "TurboQuantProd",
        method_value: str = "TurboQuantMSE",
        num_bits_key: int = 3,
        num_bits_value: int = 3,
        num_layers: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self._method_key_name = method_key
        self._method_value_name = method_value
        self._num_bits_key = num_bits_key
        self._num_bits_value = num_bits_value

        key_cls = _get_method_class(method_key)
        val_cls = _get_method_class(method_value)

        # Per-layer, per-head compressors
        self._compressors: list[list[KVCacheCompressor]] = []
        for layer in range(num_layers):
            layer_comps = []
            for head in range(num_kv_heads):
                head_seed = seed + layer * 1000 + head
                k_q = key_cls(d=head_dim, num_bits=num_bits_key, seed=head_seed)
                v_q = val_cls(d=head_dim, num_bits=num_bits_value, seed=head_seed)
                layer_comps.append(KVCacheCompressor(k_q, v_q))
            self._compressors.append(layer_comps)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress and append new KV pairs, return decompressed for attention.

        Args:
            key_states: (batch, num_kv_heads, seq_len, head_dim)
            value_states: (batch, num_kv_heads, seq_len, head_dim)
            layer_idx: Transformer layer index.

        Returns:
            (all_keys, all_values): Decompressed full cache tensors,
            shape (batch, num_kv_heads, total_seq, head_dim).
        """
        device = key_states.device
        dtype = key_states.dtype
        batch_size = key_states.shape[0]

        # Process batch=0 only for now (single-batch inference)
        k_np = key_states[0].detach().cpu().to(torch.float64).numpy()  # (heads, seq, d)
        v_np = value_states[0].detach().cpu().to(torch.float64).numpy()

        for h in range(self.num_kv_heads):
            comp = self._compressors[layer_idx][h]
            comp.compress(k_np[h], v_np[h])  # (seq, d)

        return self.get(layer_idx, device=device, dtype=dtype, batch_size=batch_size)

    def get(
        self,
        layer_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress full KV cache for a layer.

        Returns:
            (keys, values): shape (batch, num_kv_heads, total_seq, head_dim)
        """
        all_keys = []
        all_values = []
        for h in range(self.num_kv_heads):
            comp = self._compressors[layer_idx][h]
            k = comp.get_keys()   # (seq, d) numpy
            v = comp.get_values()
            all_keys.append(k)
            all_values.append(v)

        seq_len = all_keys[0].shape[0] if len(all_keys[0]) > 0 else 0
        if seq_len == 0:
            shape = (batch_size, self.num_kv_heads, 0, self.head_dim)
            k_out = torch.zeros(shape, device=device, dtype=dtype)
            v_out = torch.zeros(shape, device=device, dtype=dtype)
            return k_out, v_out

        # Stack: (heads, seq, d) → (batch, heads, seq, d)
        k_np = np.stack(all_keys)   # (heads, seq, d)
        v_np = np.stack(all_values)
        k_t = torch.from_numpy(k_np).unsqueeze(0).expand(batch_size, -1, -1, -1)
        v_t = torch.from_numpy(v_np).unsqueeze(0).expand(batch_size, -1, -1, -1)

        if dtype is not None:
            k_t = k_t.to(dtype)
            v_t = v_t.to(dtype)
        if device is not None:
            k_t = k_t.to(device)
            v_t = v_t.to(device)

        return k_t, v_t

    def seq_len(self, layer_idx: int = 0) -> int:
        """Number of cached tokens for a given layer."""
        return self._compressors[layer_idx][0].num_tokens

    def memory_bytes(self) -> int:
        """Total compressed cache size in bytes."""
        total_bits = 0
        for layer_comps in self._compressors:
            for comp in layer_comps:
                total_bits += comp.storage_bits()
        return total_bits // 8

    def compression_ratio(self) -> float:
        """Compression ratio vs fp16 storage."""
        total_tokens = sum(
            comp.num_tokens
            for layer_comps in self._compressors
            for comp in layer_comps
        )
        if total_tokens == 0:
            return 0.0
        fp16_bits = total_tokens * self.head_dim * 16 * 2  # K + V
        compressed_bits = sum(
            comp.storage_bits()
            for layer_comps in self._compressors
            for comp in layer_comps
        )
        return fp16_bits / max(compressed_bits, 1)

    def clear(self) -> None:
        """Clear all compressed caches."""
        for layer_comps in self._compressors:
            for comp in layer_comps:
                comp.clear()
