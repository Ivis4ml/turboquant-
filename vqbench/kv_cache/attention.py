"""
Attention computation on compressed KV cache.

Implements standard scaled dot-product attention using decompressed keys/values.
"""

from __future__ import annotations

import numpy as np

from vqbench.kv_cache.compressor import KVCacheCompressor


def compressed_attention(
    query: np.ndarray,
    compressor: KVCacheCompressor,
    scale: float | None = None,
) -> np.ndarray:
    """
    Scaled dot-product attention on compressed KV cache.

    Formula: softmax(Q · K^T / √d) · V

    Args:
        query: Query vector(s), shape (head_dim,) or (n_queries, head_dim).
        compressor: KVCacheCompressor holding compressed K, V.
        scale: Attention scale factor. Default: 1/√head_dim.

    Returns:
        Attention output, same shape as query.
    """
    if compressor.num_tokens == 0:
        if query.ndim == 1:
            return np.zeros(compressor.head_dim)
        return np.zeros_like(query)

    d = compressor.head_dim
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    keys = compressor.get_keys()      # (n_tokens, d)
    values = compressor.get_values()   # (n_tokens, d)

    single = query.ndim == 1
    if single:
        query = query[np.newaxis, :]   # (1, d)

    # Attention scores: Q · K^T / √d
    scores = query @ keys.T * scale    # (n_q, n_tokens)

    # Numerical stability: subtract max before softmax
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-30

    # Output: weighted sum of values
    output = weights @ values           # (n_q, d)

    if single:
        return output[0]
    return output


def attention_mse(
    queries: np.ndarray,
    keys_full: np.ndarray,
    values_full: np.ndarray,
    compressor: KVCacheCompressor,
) -> float:
    """
    Measure MSE between full-precision and compressed attention output.

    Args:
        queries: Shape (n_q, d).
        keys_full: Full precision keys, shape (n_tokens, d).
        values_full: Full precision values, shape (n_tokens, d).
        compressor: Compressed KV cache.

    Returns:
        Average MSE per query.
    """
    d = queries.shape[1]
    scale = 1.0 / np.sqrt(d)

    # Full precision attention
    scores_full = queries @ keys_full.T * scale
    scores_full -= scores_full.max(axis=1, keepdims=True)
    weights_full = np.exp(scores_full)
    weights_full /= weights_full.sum(axis=1, keepdims=True) + 1e-30
    output_full = weights_full @ values_full

    # Compressed attention
    output_compressed = compressed_attention(queries, compressor, scale)

    # MSE
    return float(np.mean(np.sum((output_full - output_compressed) ** 2, axis=1)))
