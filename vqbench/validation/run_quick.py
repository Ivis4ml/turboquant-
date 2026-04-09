#!/usr/bin/env python3
"""
Quick validation: Qwen2.5-3B (or any small model) with VQBench methods.

Phase 9.0.7 — smoke test before committing to 27B runs.

Usage:
    python -m vqbench.validation.run_quick
    python -m vqbench.validation.run_quick --model Qwen/Qwen2.5-3B --bits 3
    python -m vqbench.validation.run_quick --model Qwen/Qwen2.5-0.5B --bits 2,3,4 --max-tokens 2048
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch


def make_tq_mse_factory(bits):
    """Factory: TurboQuantMSE at given bits."""
    def factory(d, seed=42):
        from vqbench.methods.turboquant.mse import TurboQuantMSE
        return TurboQuantMSE(d=d, num_bits=bits, seed=seed, norm_correction=True)
    return factory


def make_tq_prod_factory(bits):
    """Factory: TurboQuantProd at given bits."""
    def factory(d, seed=42):
        from vqbench.methods.turboquant.prod import TurboQuantProd
        return TurboQuantProd(d=d, num_bits=bits, seed=seed, norm_correction=True)
    return factory


def make_rabitq_factory(bits):
    """Factory: ExtRaBitQ at given bits."""
    def factory(d, seed=42):
        from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
        q = ExtRaBitQ(d=d, num_bits=bits, seed=seed)
        return q
    return factory


def main():
    parser = argparse.ArgumentParser(description="VQBench quick validation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model ID")
    parser.add_argument("--bits", default="3",
                        help="Comma-separated bit-widths to test (e.g., '2,3,4')")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens from WikiText-2")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="PPL sliding window size")
    parser.add_argument("--stride", type=int, default=512,
                        help="PPL sliding window stride")
    parser.add_argument("--k-mse-only", action="store_true",
                        help="Skip PPL, only measure K-MSE (fast)")
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Bits:   {bits_list}")
    print()

    # ---- Load model ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", end=" ", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
    print(f"done ({time.time() - t0:.1f}s)")

    config = model.config
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    num_layers = config.num_hidden_layers
    print(f"Architecture: head_dim={head_dim}, num_kv_heads={num_kv_heads}, "
          f"num_layers={num_layers}")
    print()

    # ---- Load WikiText-2 ----
    from vqbench.datasets.wikitext import load_wikitext2_encodings

    print(f"Loading WikiText-2 (max {args.max_tokens} tokens)...", end=" ", flush=True)
    encodings = load_wikitext2_encodings(tokenizer, max_tokens=args.max_tokens)
    print(f"done ({encodings.size(1)} tokens)")
    print()

    # ---- Build configs ----
    # Default: K-only quantization (V left at fp16), matching turboquant_plus.
    # K-only is the proven approach: V quantization can be added separately.
    configs = []
    for b in bits_list:
        configs.append((f"TQ-MSE K{b}b", make_tq_mse_factory(b), None))
        if b >= 2:
            configs.append((f"TQ-Prod K{b}b", make_tq_prod_factory(b), None))
        if b >= 2:
            configs.append((f"RaBitQ K{b}b", make_rabitq_factory(b), None))
        # Also test K+V for comparison
        configs.append((f"TQ-MSE KV{b}b", make_tq_mse_factory(b), make_tq_mse_factory(b)))

    # ==================================================================
    # Phase 1: Normalized K-cache MSE on real model activations
    # ==================================================================
    print("=" * 65)
    print("PHASE 1: Normalized K-cache MSE (primary quality metric)")
    print("=" * 65)

    mse_results = {}
    for name, k_factory, v_factory in configs:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        norm_mse = _measure_normalized_k_mse(model, encodings, k_factory, device,
                                              max_chunks=8, chunk_size=256)
        elapsed = time.time() - t0
        mse_results[name] = norm_mse
        print(f"norm_MSE={norm_mse:.6f}  ({elapsed:.1f}s)")

    if args.k_mse_only:
        _print_summary(mse_results, {}, 0.0)
        return

    # ==================================================================
    # Phase 2: Perplexity with monkey-patched K+V
    # ==================================================================
    print()
    print()
    print("=" * 65)
    print("PHASE 2: Perplexity (monkey-patched K quantization)")
    print("  NOTE: This is an UPPER BOUND. Real KV cache compression only")
    print("  quantizes PAST tokens; current token uses exact K. Monkey-patching")
    print("  quantizes ALL positions, which is significantly worse.")
    print("=" * 65)

    from vqbench.validation.ppl_eval import evaluate_ppl
    from vqbench.validation.monkey_patch import patch_model_kv, unpatch_model

    ppl_results = {}

    # Baseline
    print("  fp16 baseline...", end=" ", flush=True)
    t0 = time.time()
    ppl, n_tok = evaluate_ppl(model, tokenizer, encodings, device,
                               max_length=args.max_length, stride=args.stride)
    ppl_results["fp16"] = ppl
    print(f"PPL={ppl:.2f}  ({n_tok} tokens, {time.time() - t0:.1f}s)")

    # Quantized variants
    for name, k_factory, v_factory in configs:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        hooks = patch_model_kv(model, k_quantizer_factory=k_factory,
                               v_quantizer_factory=v_factory)
        try:
            ppl, n_tok = evaluate_ppl(model, tokenizer, encodings, device,
                                       max_length=args.max_length, stride=args.stride)
        finally:
            unpatch_model(hooks)
        ppl_results[name] = ppl
        print(f"PPL={ppl:.2f}  ({n_tok} tokens, {time.time() - t0:.1f}s)")

    # ==================================================================
    # Summary
    # ==================================================================
    baseline_ppl = ppl_results.get("fp16", 0.0)
    _print_summary(mse_results, ppl_results, baseline_ppl)


@torch.no_grad()
def _measure_normalized_k_mse(model, encodings, quantizer_factory, device,
                               max_chunks=8, chunk_size=256):
    """Measure ‖K - K̂‖² / ‖K‖² on real model K tensors (normalized MSE)."""
    seq_len = encodings.size(1)
    total_mse = 0.0
    total_count = 0

    for i in range(min(max_chunks, seq_len // chunk_size)):
        begin = i * chunk_size
        end = begin + chunk_size
        input_ids = encodings[:, begin:end].to(device)

        outputs = model(input_ids, use_cache=True)
        cache = outputs.past_key_values

        for layer in cache.layers:
            k = layer.keys  # (batch, heads, seq, head_dim)
            B, H, S, D = k.shape
            for b in range(B):
                for h in range(H):
                    k_np = k[b, h].float().cpu().numpy()
                    q = quantizer_factory(D, seed=h)
                    k_hat = q.dequantize_batch(q.quantize_batch(k_np))
                    # Normalized MSE: ‖K - K̂‖² / ‖K‖²
                    norms_sq = np.sum(k_np ** 2, axis=1)
                    diff_sq = np.sum((k_np - k_hat) ** 2, axis=1)
                    norm_mse = np.mean(diff_sq / np.maximum(norms_sq, 1e-30))
                    total_mse += norm_mse
                    total_count += 1

    return total_mse / total_count if total_count > 0 else float("inf")


def _print_summary(mse_results, ppl_results, baseline_ppl):
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    has_ppl = len(ppl_results) > 0
    if has_ppl:
        header = f"  {'Method':<18s}  {'PPL':>8s}  {'ΔPPL':>8s}  {'norm_MSE':>10s}"
    else:
        header = f"  {'Method':<18s}  {'norm_MSE':>10s}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    if has_ppl and "fp16" in ppl_results:
        print(f"  {'fp16 baseline':<18s}  {baseline_ppl:>8.2f}  {'—':>8s}  {'—':>12s}")

    all_names = list(mse_results.keys())
    for name in all_names:
        mse_str = f"{mse_results[name]:.6f}"
        if has_ppl and name in ppl_results:
            ppl_val = ppl_results[name]
            delta = ppl_val - baseline_ppl
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            print(f"  {name:<18s}  {ppl_val:>8.2f}  {delta_str:>8s}  {mse_str:>12s}")
        else:
            if has_ppl:
                print(f"  {name:<18s}  {'—':>8s}  {'—':>8s}  {mse_str:>12s}")
            else:
                print(f"  {name:<18s}  {mse_str:>12s}")

    print()


if __name__ == "__main__":
    main()
