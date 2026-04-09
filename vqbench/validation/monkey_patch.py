"""
Monkey-patch K-proj and V-proj to inject quantize→dequantize.

Proven approach from turboquant_plus: wrap the linear projection so the
quantization error is injected directly into the forward pass. This measures
the EXACT effect of KV compression on attention and perplexity.

Works with any HuggingFace model using standard attention (Qwen2, Gemma2, LLaMA, Mistral).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn


def quant_dequant_tensor(
    states: torch.Tensor,
    quantizer_factory: Callable,
) -> torch.Tensor:
    """
    Quantize-dequantize a KV tensor through a given quantizer.

    Args:
        states: (batch, n_heads, seq_len, head_dim) float tensor.
        quantizer_factory: callable(d, seed) → quantizer with quantize_batch/dequantize_batch.

    Returns:
        Quantize-dequantized tensor, same shape and dtype.
    """
    B, H, S, D = states.shape
    device = states.device
    dtype = states.dtype

    out = torch.empty_like(states)
    for b in range(B):
        for h in range(H):
            k_np = states[b, h].float().cpu().numpy()  # (S, D)
            q = quantizer_factory(D, seed=h)
            qvs = q.quantize_batch(k_np)
            k_hat = q.dequantize_batch(qvs)
            out[b, h] = torch.from_numpy(k_hat).to(dtype=dtype, device=device)
    return out


class _QuantizedProj(nn.Module):
    """Wraps a linear projection to inject quantize→dequantize after it."""

    def __init__(self, orig: nn.Module, quant_factory: Callable,
                 head_dim: int, num_kv_heads: int):
        super().__init__()
        self.orig = orig
        self.qf = quant_factory
        self.hd = head_dim
        self.nkv = num_kv_heads

    def forward(self, x, *args, **kwargs):
        out = self.orig(x, *args, **kwargs)
        B, S, _ = out.shape
        reshaped = out.view(B, S, self.nkv, self.hd).permute(0, 2, 1, 3)
        quantized = quant_dequant_tensor(reshaped, self.qf)
        return quantized.permute(0, 2, 1, 3).reshape(B, S, -1)


def patch_model_kv(
    model,
    k_quantizer_factory: Callable | None = None,
    v_quantizer_factory: Callable | None = None,
) -> list[tuple]:
    """
    Patch all attention layers to quantize K and/or V before caching.

    Args:
        model: HuggingFace CausalLM model.
        k_quantizer_factory: callable(d, seed) → quantizer for keys. None = skip K.
        v_quantizer_factory: callable(d, seed) → quantizer for values. None = skip V.

    Returns:
        List of (attn_module, original_k_proj, original_v_proj) for unpatching.
    """
    config = model.config
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    hooks = []

    # Find attention layers — supports Qwen2, LLaMA, Mistral, Gemma2
    layers = _get_model_layers(model)

    for layer in layers:
        attn = _get_attn_module(layer)
        if attn is None:
            continue

        orig_k = attn.k_proj
        orig_v = attn.v_proj

        if k_quantizer_factory is not None:
            attn.k_proj = _QuantizedProj(orig_k, k_quantizer_factory, head_dim, num_kv_heads)
        if v_quantizer_factory is not None:
            attn.v_proj = _QuantizedProj(orig_v, v_quantizer_factory, head_dim, num_kv_heads)

        hooks.append((attn, orig_k, orig_v))

    return hooks


def unpatch_model(hooks: list[tuple]) -> None:
    """Restore original projections."""
    for attn, orig_k, orig_v in hooks:
        attn.k_proj = orig_k
        attn.v_proj = orig_v


def _get_model_layers(model):
    """Extract transformer layers from various HF model architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported model architecture: {type(model).__name__}")


def _get_attn_module(layer):
    """Extract attention module from a transformer layer."""
    for attr in ["self_attn", "attention", "attn"]:
        if hasattr(layer, attr):
            mod = getattr(layer, attr)
            if hasattr(mod, "k_proj") and hasattr(mod, "v_proj"):
                return mod
    return None
