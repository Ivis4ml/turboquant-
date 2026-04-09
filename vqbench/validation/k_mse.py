"""
Measure MSE of quantized K tensors on real model activations.

Runs forward passes, extracts real K tensors from past_key_values,
applies quantize→dequantize, measures MSE. Fast and model-agnostic.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from vqbench.validation.monkey_patch import quant_dequant_tensor


@torch.no_grad()
def measure_k_mse(
    model,
    encodings: torch.Tensor,
    quantizer_factory: Callable,
    device: str | torch.device = "cpu",
    max_chunks: int = 10,
    chunk_size: int = 256,
) -> float:
    """
    Measure average MSE of quantized K vs original K on real activations.

    Args:
        model: HuggingFace CausalLM.
        encodings: Token IDs, shape (1, seq_len).
        quantizer_factory: callable(d, seed) → quantizer.
        device: Compute device.
        max_chunks: Max number of chunks to process.
        chunk_size: Tokens per chunk.

    Returns:
        Average K-cache MSE across all layers and chunks.
    """
    seq_len = encodings.size(1)
    total_mse = 0.0
    total_count = 0

    for i in range(min(max_chunks, seq_len // chunk_size)):
        begin = i * chunk_size
        end = begin + chunk_size
        input_ids = encodings[:, begin:end].to(device)

        outputs = model(input_ids, output_attentions=False, use_cache=True)
        past_kv = outputs.past_key_values

        # past_kv is a tuple of (key, value) per layer
        for layer_kv in past_kv:
            if isinstance(layer_kv, (tuple, list)):
                k = layer_kv[0]  # (batch, n_heads, seq_len, head_dim)
            else:
                # Newer transformers: DynamicCache object
                continue

            k_quant = quant_dequant_tensor(k, quantizer_factory)
            mse = ((k.float().cpu() - k_quant.float().cpu()) ** 2).mean().item()
            total_mse += mse
            total_count += 1

    return total_mse / total_count if total_count > 0 else float("inf")
