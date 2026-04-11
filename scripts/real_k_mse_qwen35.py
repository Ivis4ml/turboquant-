"""
Real K-cache MSE on Qwen3.5-4B activations.

Reproduces REPORT.md §3 (historical §3.2) but on the Qwen3.5 target model:
extracts real K tensors from the 8 full-attention layers via forward hooks,
applies each VQBench quantizer offline, reports normalized reconstruction
MSE averaged over (layer × head × chunk).

Unlike scripts/reproduce_qwen35_4b_headline.py this script does NOT perturb
the forward pass — it only measures the quantizer's distortion on realistic
K-cache activation statistics. It is the right tool when you want to compare
quantizer quality on real data without the PPL noise floor.

Hybrid attention handling: the 24 linear-attention layers do not have k_proj
and are skipped automatically. Only the 8 full-attention layers contribute.

Requires [validation] extras. Runtime ~60 s on M5 Pro with cached weights.

Usage:
    python scripts/real_k_mse_qwen35.py                              # single-seed
    python scripts/real_k_mse_qwen35.py --seeds 42 43 44             # variance
    python scripts/real_k_mse_qwen35.py --bits 2 3 4 --max-tokens 2048
    python scripts/real_k_mse_qwen35.py --model Qwen/Qwen3-4B        # head_dim=128
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="token budget (more → more K samples, slower)")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="rotation seeds (pass multiple for mean±std reporting)")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[32, 64],
                        help="BlockTurboQuantMSE block sizes to test")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "This script needs the [validation] extras. "
            "Run: pip install -e '.[test,validation]'"
        ) from e

    from vqbench.core.metrics import mse_distortion
    from vqbench.datasets.wikitext import load_wikitext2_encodings
    from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
    from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
    from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
    from vqbench.methods.turboquant.mse import TurboQuantMSE
    from vqbench.methods.turboquant.prod import TurboQuantProd

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
    print(f"Config: head_dim={head_dim}, num_kv_heads={num_kv_heads}")

    # Register forward hooks on every k_proj we can find
    captured: list[np.ndarray] = []  # list of (n_kv_heads, seq, head_dim)

    def _make_hook(layer_idx: int):
        def hook(module, inp, out):
            # out: (batch, seq, n_kv_heads * head_dim)
            B, S, _ = out.shape
            if B != 1:
                return
            reshaped = out.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            # → (1, n_kv_heads, seq, head_dim)
            captured.append(reshaped[0].detach().float().cpu().numpy())
        return hook

    hooks = []
    n_layers_hooked = 0
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    for i, layer in enumerate(layers):
        attn = None
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break
        if attn is not None and hasattr(attn, "k_proj"):
            h = attn.k_proj.register_forward_hook(_make_hook(i))
            hooks.append(h)
            n_layers_hooked += 1
    print(f"Hooked k_proj on {n_layers_hooked} full-attention layers")

    # Forward pass to fill the captured list
    print(f"Running forward pass on WikiText-2 first {args.max_tokens} tokens ...")
    enc = load_wikitext2_encodings(tok, max_tokens=args.max_tokens)
    n_chunks = enc.size(1) // args.chunk_size
    with torch.no_grad():
        for i in range(n_chunks):
            chunk_ids = enc[:, i * args.chunk_size : (i + 1) * args.chunk_size].to(args.device)
            _ = model(chunk_ids, use_cache=False)

    for h in hooks:
        h.remove()

    # Stack: each capture is (n_kv_heads, seq, d). We got n_chunks * n_layers_hooked of these.
    # Reshape to (N, d) where N = total (layer, head, seq) triples.
    all_k = []
    for cap in captured:
        nkv, seq, d = cap.shape
        all_k.append(cap.reshape(-1, d))
    K = np.concatenate(all_k, axis=0)  # (N, d)
    print(f"Collected {K.shape[0]} real K vectors of dim {K.shape[1]}")

    # For each method × bit, quantize and measure nMSE. Each (method, bit) cell
    # is aggregated over all args.seeds: single seed → just the value, multiple
    # seeds → mean ± std.
    def fit_quantize(q, X: np.ndarray) -> np.ndarray:
        if hasattr(q, "fit"):
            q.fit(X)
        qvs = q.quantize_batch(X)
        return q.dequantize_batch(qvs)

    d = K.shape[1]

    def run_method_multi_seed(label: str, q_factory: Callable[[int, int], object]) -> dict:
        """Returns {'method': label, 'b{b}': [list of nMSE across seeds]}."""
        row: dict = {"method": label}
        for b in args.bits:
            values: list[float] = []
            for seed in args.seeds:
                try:
                    q = q_factory(b, seed)
                except (ValueError, TypeError):
                    continue
                if q is None:
                    continue
                try:
                    K_hat = fit_quantize(q, K)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{label} b={b} seed={seed}] FAILED: {e}")
                    continue
                values.append(mse_distortion(K, K_hat))
            row[f"b{b}"] = values
        return row

    results = []
    print(f"\nQuantizing with {len(args.seeds)} seed(s): {args.seeds} ...")
    results.append(run_method_multi_seed(
        "TurboQuantMSE",
        lambda b, s: TurboQuantMSE(d=d, num_bits=b, seed=s, norm_correction=True),
    ))
    results.append(run_method_multi_seed(
        "TurboQuantProd",
        lambda b, s: TurboQuantProd(d=d, num_bits=b, seed=s, norm_correction=True),
    ))
    for bs in args.block_sizes:
        if d % bs != 0:
            continue
        results.append(run_method_multi_seed(
            f"BlockTQ-B{bs}",
            lambda b, s, _bs=bs: BlockTurboQuantMSE(
                d=d, num_bits=b, block_size=_bs, seed=s, norm_correction=True,
            ),
        ))
    if 1 in args.bits:
        rq1_row: dict = {"method": "RaBitQ 1-bit", **{f"b{b}": [] for b in args.bits}}
        rq1_vals = []
        for seed in args.seeds:
            q = RaBitQ1Bit(d=d, seed=seed)
            q.fit(K)
            K_hat = q.dequantize_batch(q.quantize_batch(K))
            rq1_vals.append(mse_distortion(K, K_hat))
        rq1_row["b1"] = rq1_vals
        results.append(rq1_row)
    results.append(run_method_multi_seed(
        "ExtRaBitQ",
        lambda b, s: ExtRaBitQ(d=d, num_bits=b, seed=s) if b >= 2 else None,
    ))

    # Print table
    n_seeds = len(args.seeds)
    mode_tag = "mean ± std" if n_seeds > 1 else "single-seed"
    print(f"\n=== Real K-cache nMSE on {args.model} ({mode_tag}, {n_seeds} seed(s)) ===")
    print(f"    head_dim={d}, {K.shape[0]} K vectors, {n_layers_hooked} full-attention layers")

    col_width = 18 if n_seeds > 1 else 12
    header = f"{'method':<18s}"
    for b in args.bits:
        header += f" {f'b={b}':>{col_width}s}"
    print(header)
    print("-" * len(header))

    def fmt(values: list[float]) -> str:
        if not values:
            return f"{'—':>{col_width}s}"
        if len(values) == 1:
            return f"{values[0]:>{col_width}.6f}"
        mean = float(np.mean(values))
        std = float(np.std(values))
        return f"{mean:>9.6f}±{std:.5f}"

    for row in results:
        line = f"{row['method']:<18s}"
        for b in args.bits:
            vals = row.get(f"b{b}") or []
            line += f" {fmt(vals):>{col_width}s}"
        print(line)

    print("\nNotes:")
    print("  * nMSE = E[||k - k_hat||^2 / ||k||^2] over real K vectors.")
    print("  * Lower is better. TurboQuantMSE should be near the Lloyd-Max bound")
    print("    (~0.009 at b=4); deviations reveal how non-Gaussian the real K")
    print("    distribution is after Haar rotation.")
    print("  * This is an OFFLINE measurement — the model forward pass is not")
    print("    perturbed. For end-to-end PPL see reproduce_qwen35_4b_headline.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
