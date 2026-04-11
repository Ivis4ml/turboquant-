"""
Pluggable KV-cache compressor: any VQ method for K, any for V.

Req F6: Pluggable KV-cache compressor: any VQ method for K, any for V.
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector


class KVCacheCompressor:
    """
    Compress KV cache using any pair of VectorQuantizer methods.

    Supports different methods/bit-widths for keys vs values
    (keys need unbiased IP for attention; values need low MSE for output).

    Args:
        key_quantizer: VectorQuantizer for compressing keys.
        value_quantizer: VectorQuantizer for compressing values.
    """

    def __init__(
        self,
        key_quantizer: VectorQuantizer,
        value_quantizer: VectorQuantizer,
    ) -> None:
        self.key_q = key_quantizer
        self.value_q = value_quantizer
        self._compressed_keys: list[QuantizedVector] = []
        self._compressed_values: list[QuantizedVector] = []

    @property
    def num_tokens(self) -> int:
        return len(self._compressed_keys)

    @property
    def head_dim(self) -> int:
        return self.key_q.d

    def compress(self, keys: np.ndarray, values: np.ndarray) -> None:
        """
        Compress a batch of KV pairs and append to cache.

        Args:
            keys: Shape (n_new, head_dim) — new key vectors.
            values: Shape (n_new, head_dim) — new value vectors.
        """
        self._compressed_keys.extend(self.key_q.quantize_batch(keys))
        self._compressed_values.extend(self.value_q.quantize_batch(values))

    def get_keys(self) -> np.ndarray:
        """Decompress all keys. Shape (n_tokens, head_dim)."""
        if not self._compressed_keys:
            return np.empty((0, self.head_dim))
        return self.key_q.dequantize_batch(self._compressed_keys)

    def get_values(self) -> np.ndarray:
        """Decompress all values. Shape (n_tokens, head_dim)."""
        if not self._compressed_values:
            return np.empty((0, self.head_dim))
        return self.value_q.dequantize_batch(self._compressed_values)

    def storage_bits(self) -> int:
        """Total bits used by the compressed cache."""
        k_bits = sum(self.key_q.storage_bits(qv) for qv in self._compressed_keys)
        v_bits = sum(self.value_q.storage_bits(qv) for qv in self._compressed_values)
        return k_bits + v_bits

    def compression_ratio(self) -> float:
        """
        Compression ratio vs fp16 storage.

        fp16 is the reference baseline for KV-cache reporting because that is
        what HuggingFace transformers and llama.cpp actually store. The earlier
        implementation reported against fp32, which double-counted the savings
        and was inconsistent with QuantizedKVCache.compression_ratio() /
        VQBenchCache.compression_ratio(). All three paths now agree.
        """
        n = self.num_tokens
        if n == 0:
            return 0.0
        fp16_bits = n * self.head_dim * 16 * 2  # keys + values
        return fp16_bits / max(self.storage_bits(), 1)

    def clear(self) -> None:
        """Clear the compressed cache."""
        self._compressed_keys.clear()
        self._compressed_values.clear()
