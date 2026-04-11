"""
Diagnostic: why does 4-bit K on Qwen3.5-4B give strictly 0.00% ΔPPL?

Answers four sub-questions:

  1. Are the monkey-patch hooks actually firing on all 8 full-attention layers?
  2. Is the quantized K tensor actually different from the original K?
     (i.e. is monkey-patch producing a real perturbation on the forward path?)
  3. Is the PPL difference truly zero, or just rounded to 4 decimals?
     (manual cross-entropy in fp32, 10-decimal print)
  4. How does per-layer K-MSE flow through the architecture?

Useful when the §3 "strictly lossless" result looks too clean.

Usage:
    python scripts/diagnose_qwen35_lossless.py
    python scripts/diagnose_qwen35_lossless.py --bits 4 --method TurboQuantMSE
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--method", default="TurboQuantMSE")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Needs [validation] extras. Run: pip install -e '.[test,validation]'"
        ) from e

    from vqbench.datasets.wikitext import load_wikitext2_encodings
    from vqbench.validation.monkey_patch import patch_model_kv
    from vqbench.validation.streaming_ppl import evaluate_streaming_ppl

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(args.device).eval()

    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads,
    )
    print(f"  head_dim={head_dim}, num_kv_heads={num_kv_heads}")

    enc = load_wikitext2_encodings(tok, max_tokens=args.max_tokens)

    # --- Q1: count hooked layers and hook fire counts ---
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    full_attn_layer_indices = []
    for i, layer in enumerate(layers):
        attn = None
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break
        if attn is not None and hasattr(attn, "k_proj"):
            full_attn_layer_indices.append(i)
    print(f"\n[Q1] Full-attention layers (with k_proj): "
          f"{len(full_attn_layer_indices)}/{len(layers)}")
    print(f"     Layer indices: {full_attn_layer_indices}")

    # Instrument hooks to count firings + capture pre/post tensors
    fire_counts: dict[int, int] = {i: 0 for i in full_attn_layer_indices}
    sample_pre_post: dict[int, tuple] = {}  # layer_idx -> (K_before, K_after)

    from vqbench.methods.turboquant.mse import TurboQuantMSE
    from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
    from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ

    def factory(d: int, seed: int = 0):
        actual_seed = args.seed * 1000 + seed
        if args.method == "TurboQuantMSE":
            return TurboQuantMSE(d=d, num_bits=args.bits, seed=actual_seed, norm_correction=True)
        if args.method == "BlockTurboQuantMSE":
            return BlockTurboQuantMSE(d=d, num_bits=args.bits, block_size=64,
                                      seed=actual_seed, norm_correction=True)
        if args.method == "ExtRaBitQ":
            return ExtRaBitQ(d=d, num_bits=args.bits, seed=actual_seed)
        raise ValueError(args.method)

    # Instrument each k_proj with a pre/post-capture hook
    from vqbench.validation.monkey_patch import quant_dequant_tensor

    class _InstrumentedQuantizedProj(torch.nn.Module):
        def __init__(self, orig, layer_idx):
            super().__init__()
            self.orig = orig
            self.layer_idx = layer_idx

        def forward(self, x, *args_, **kwargs_):
            out = self.orig(x, *args_, **kwargs_)
            B, S, _ = out.shape
            reshaped = out.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            quantized = quant_dequant_tensor(reshaped, factory)
            fire_counts[self.layer_idx] += 1
            if self.layer_idx not in sample_pre_post:
                sample_pre_post[self.layer_idx] = (
                    reshaped.detach().float().cpu().numpy().copy(),
                    quantized.detach().float().cpu().numpy().copy(),
                )
            return quantized.permute(0, 2, 1, 3).reshape(B, S, -1)

    # Patch manually
    patched = []
    for idx in full_attn_layer_indices:
        layer = layers[idx]
        attn = None
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break
        orig_k = attn.k_proj
        attn.k_proj = _InstrumentedQuantizedProj(orig_k, idx)
        patched.append((attn, orig_k, idx))

    # Run the forward pass
    print(f"\n[Q2/Q3] Running {args.method} {args.bits}-bit K quantized forward pass ...")
    ppl_q, _ = evaluate_streaming_ppl(
        model, enc, device=args.device, chunk=args.chunk, cache=None,
    )
    # Unpatch and rerun for baseline
    for attn, orig_k, _ in patched:
        attn.k_proj = orig_k
    ppl_base, _ = evaluate_streaming_ppl(
        model, enc, device=args.device, chunk=args.chunk, cache=None,
    )

    print(f"\n[Q1] Hook fire counts per layer:")
    for idx, c in fire_counts.items():
        print(f"     layer {idx:2d}: {c} calls")
    total_fires = sum(fire_counts.values())
    print(f"     total:     {total_fires} hook calls across {len(fire_counts)} layers")

    print(f"\n[Q2] Quantization perturbation per layer (first chunk):")
    print(f"     {'layer':>6s} {'||K||':>12s} {'||K-K_hat||':>14s} {'nMSE':>10s} "
          f"{'max|K-K_hat|':>14s}")
    for idx, (K_before, K_after) in sorted(sample_pre_post.items()):
        diff = K_before - K_after
        k_norm = float(np.linalg.norm(K_before))
        diff_norm = float(np.linalg.norm(diff))
        nmse = (diff_norm / k_norm) ** 2 if k_norm > 0 else 0.0
        max_abs = float(np.max(np.abs(diff)))
        print(f"     {idx:>6d} {k_norm:>12.4f} {diff_norm:>14.4f} {nmse:>10.6f} "
              f"{max_abs:>14.6f}")

    print(f"\n[Q3] PPL precision check:")
    print(f"     baseline PPL = {ppl_base:.10f}")
    print(f"     quant    PPL = {ppl_q:.10f}")
    print(f"     raw diff     = {ppl_q - ppl_base:+.10f}")
    print(f"     rounded (4)  = {ppl_q - ppl_base:+.4f}")
    if abs(ppl_q - ppl_base) < 1e-6:
        print(f"     → PPL change is below fp32 accumulation noise")
    elif abs(ppl_q - ppl_base) < 1e-3:
        print(f"     → PPL change is below 4-decimal display precision")
    else:
        print(f"     → PPL change is visible at 4-decimal precision")

    print(f"\n[Q4] Summary:")
    print(f"     - {len(full_attn_layer_indices)} full-attention layers are hooked.")
    if total_fires > 0:
        print(f"     - Hooks fired {total_fires} times during {args.max_tokens}-token eval.")
    if sample_pre_post:
        # Average nMSE across layers
        nmses = []
        for K_before, K_after in sample_pre_post.values():
            k_norm = np.linalg.norm(K_before)
            diff_norm = np.linalg.norm(K_before - K_after)
            nmses.append((diff_norm / k_norm) ** 2 if k_norm > 0 else 0.0)
        print(f"     - Average per-layer K nMSE = {np.mean(nmses):.6f}")
        print(f"       Expected for {args.method} at b={args.bits}: ~{4**-args.bits:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
