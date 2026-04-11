"""
Inner-product bias factor α on Qwen3.5-4B real K activations.

Reproduces REPORT.md historical §3.3 on the Qwen3.5 target model: extracts
real K tensors from Qwen3.5-4B full-attention layers, quantizes them with
each VQBench method, and fits

    E[⟨q, k_hat⟩] ≈ α · ⟨q, k⟩

by least squares on paired (query, key) samples. Reports α per (method,
bit-width). α = 1 means unbiased (no systematic shrinkage of attention
logits after softmax); α < 1 means the attention distribution gets
multiplicatively flattened.

Two paths are reported where meaningful:

    "direct"     — α computed from dequantized k (what a standard
                   HuggingFace attention module sees).
    "estimator"  — α computed using the method-specific unbiased estimator
                   (rabitq_ip_estimate, ext_rabitq_ip_estimate) which is
                   the paper-claimed unbiasedness path.

Queries are synthetic random unit vectors at d = head_dim — the point of
the table is to characterize the quantizer's bias on realistic K statistics,
not the query distribution.

Requires [validation] extras. Runtime ~90 s on M5 Pro with cached weights.

Usage:
    python scripts/ip_bias_alpha_qwen35.py
    python scripts/ip_bias_alpha_qwen35.py --bits 1 2 3 4
    python scripts/ip_bias_alpha_qwen35.py --n-keys 5000 --n-queries 1000
"""

from __future__ import annotations

import argparse

import numpy as np


def _fit_alpha_from_estimator(
    K: np.ndarray, Q: np.ndarray, qvs: list, estimator_fn, centroid: np.ndarray
) -> float:
    """Fit α for a RaBitQ-style estimator that outputs ⟨q - c, k - c⟩.

    α = sum(true * est) / sum(true^2) where "true" is the centered IP the
    estimator targets. This is NOT the same as the direct-path α, which
    fits ⟨q, k⟩; on data with non-zero centroid (e.g. real K activations)
    the two quantities differ. Report both separately in the table.
    """
    n = len(K)
    n_q = len(Q)
    true_ips = np.zeros(n)
    est_ips = np.zeros(n)
    for i in range(n):
        q = Q[i % n_q]
        # Estimator targets ⟨q - c, k - c⟩, so the paired "true" value
        # must also be centered — otherwise we compare apples to oranges.
        true_ips[i] = float(np.dot(q - centroid, K[i] - centroid))
        est_ips[i] = estimator_fn(q, qvs[i])
    denom = float(np.sum(true_ips ** 2))
    if denom < 1e-30:
        return 1.0
    return float(np.sum(true_ips * est_ips) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--n-keys", type=int, default=4000,
                        help="subsample this many K vectors to keep runtime manageable")
    parser.add_argument("--n-queries", type=int, default=2000,
                        help="number of synthetic random unit-vector queries")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data-seed", type=int, default=0, help="query / subsample seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="rotation seeds (pass multiple for mean±std reporting)")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "This script needs the [validation] extras. "
            "Run: pip install -e '.[test,validation]'"
        ) from e

    from vqbench.core.metrics import ip_bias
    from vqbench.datasets.synthetic import random_unit_vectors
    from vqbench.datasets.wikitext import load_wikitext2_encodings
    from vqbench.methods.rabitq.estimator import (
        ext_rabitq_ip_estimate,
        rabitq_ip_estimate,
    )
    from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
    from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
    from vqbench.methods.turboquant.mse import TurboQuantMSE
    from vqbench.methods.turboquant.prod import TurboQuantProd

    # --- load model, extract K ---
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

    captured: list[np.ndarray] = []

    def _make_hook():
        def hook(module, inp, out):
            B, S, _ = out.shape
            if B != 1:
                return
            reshaped = out.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            captured.append(reshaped[0].detach().float().cpu().numpy())
        return hook

    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    for layer in layers:
        attn = None
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break
        if attn is not None and hasattr(attn, "k_proj"):
            hooks.append(attn.k_proj.register_forward_hook(_make_hook()))

    enc = load_wikitext2_encodings(tok, max_tokens=args.max_tokens)
    n_chunks = enc.size(1) // args.chunk_size
    with torch.no_grad():
        for i in range(n_chunks):
            chunk_ids = enc[:, i * args.chunk_size : (i + 1) * args.chunk_size].to(args.device)
            _ = model(chunk_ids, use_cache=False)
    for h in hooks:
        h.remove()

    all_k = [cap.reshape(-1, cap.shape[-1]) for cap in captured]
    K_full = np.concatenate(all_k, axis=0)
    rng = np.random.default_rng(args.data_seed)
    idx = rng.choice(K_full.shape[0], size=min(args.n_keys, K_full.shape[0]), replace=False)
    K = K_full[idx]
    d = K.shape[1]
    Q = random_unit_vectors(args.n_queries, d, seed=args.data_seed + 1)
    print(f"Using {K.shape[0]} real K vectors (d={d}), {Q.shape[0]} synthetic queries")

    # --- fit α per (method × bit), aggregated over rotation seeds ---
    # Each cell is a list of α values (one per seed).
    rows: list[tuple[str, dict[int, list[float]]]] = []

    def add_row(label: str, fn):
        row: dict[int, list[float]] = {b: [] for b in args.bits}
        for b in args.bits:
            for seed in args.seeds:
                try:
                    alpha = fn(b, seed)
                except (ValueError, TypeError):
                    continue
                if alpha is None:
                    continue
                row[b].append(alpha)
        rows.append((label, row))

    # TurboQuantMSE direct
    def tq_mse(b, seed):
        q = TurboQuantMSE(d=d, num_bits=b, seed=seed, norm_correction=True)
        K_hat = q.dequantize_batch(q.quantize_batch(K))
        alpha, _ = ip_bias(K, K_hat, Q)
        return alpha
    add_row("TQ-MSE (direct)", tq_mse)

    def tq_prod(b, seed):
        q = TurboQuantProd(d=d, num_bits=b, seed=seed, norm_correction=True)
        K_hat = q.dequantize_batch(q.quantize_batch(K))
        alpha, _ = ip_bias(K, K_hat, Q)
        return alpha
    add_row("TQ-Prod (direct=est)", tq_prod)

    def ext_direct(b, seed):
        if b < 2:
            return None
        q = ExtRaBitQ(d=d, num_bits=b, seed=seed)
        q.fit(K)
        K_hat = q.dequantize_batch(q.quantize_batch(K))
        alpha, _ = ip_bias(K, K_hat, Q)
        return alpha
    add_row("ExtRaBitQ (direct)", ext_direct)

    def ext_est(b, seed):
        if b < 2:
            return None
        q = ExtRaBitQ(d=d, num_bits=b, seed=seed)
        q.fit(K)
        qvs = q.quantize_batch(K)
        centroid = q._get_centroid(d) if hasattr(q, "_get_centroid") else np.zeros(d)
        rotation = q._rotation
        return _fit_alpha_from_estimator(
            K, Q, qvs,
            lambda q_vec, qv: ext_rabitq_ip_estimate(q_vec, qv, rotation, centroid, d, b),
            centroid,
        )
    add_row("ExtRaBitQ (estimator)", ext_est)

    def rq1_direct(b, seed):
        if b != 1:
            return None
        q = RaBitQ1Bit(d=d, seed=seed)
        q.fit(K)
        K_hat = q.dequantize_batch(q.quantize_batch(K))
        alpha, _ = ip_bias(K, K_hat, Q)
        return alpha
    add_row("RaBitQ 1-bit (direct)", rq1_direct)

    def rq1_est(b, seed):
        if b != 1:
            return None
        q = RaBitQ1Bit(d=d, seed=seed)
        q.fit(K)
        qvs = q.quantize_batch(K)
        centroid = q._get_centroid(d)
        rotation = q._rotation
        return _fit_alpha_from_estimator(
            K, Q, qvs,
            lambda q_vec, qv: rabitq_ip_estimate(q_vec, qv, rotation, centroid, d),
            centroid,
        )
    add_row("RaBitQ 1-bit (estimator)", rq1_est)

    # --- print table ---
    n_seeds = len(args.seeds)
    mode_tag = "mean ± std" if n_seeds > 1 else "single-seed"
    print(f"\n=== Inner-product bias α on {args.model} ({mode_tag}, {n_seeds} seed(s)) ===")
    print(f"    head_dim={d}, real K ({K.shape[0]} vec) × synthetic Q ({Q.shape[0]} vec)")
    print(f"    α = 1 means unbiased attention logits; α < 1 means shrunk in expectation")
    col_width = 16 if n_seeds > 1 else 10
    header = f"{'method':<24s}"
    for b in args.bits:
        header += f" {f'b={b}':>{col_width}s}"
    print(header)
    print("-" * len(header))

    def fmt(values: list[float]) -> str:
        if not values:
            return f"{'—':>{col_width}s}"
        if len(values) == 1:
            return f"{values[0]:>{col_width}.4f}"
        mean = float(np.mean(values))
        std = float(np.std(values))
        return f"{mean:>9.4f}±{std:.4f}"

    for label, row in rows:
        line = f"{label:<24s}"
        for b in args.bits:
            line += f" {fmt(row.get(b, []))}"
        print(line)

    print("\nHow to read this:")
    print("  * TQ-MSE, ExtRaBitQ (direct), RaBitQ 1-bit (direct) — all biased by")
    print("    design in the direct dequant path; α converges toward 1 as b grows.")
    print("  * TQ-Prod is unbiased by construction in the direct path (QJL residual")
    print("    injected inside dequantize), so α = 1 at any b.")
    print("  * RaBitQ 1-bit (estimator) uses the rabitq_ip_estimate correction;")
    print("    α = 1 is the paper-claimed unbiasedness.")
    print("  * ExtRaBitQ (estimator) uses ext_rabitq_ip_estimate; currently")
    print("    NOT validated at α ≈ 1 (Phase 10.2 in PLAN.md). If the entry")
    print("    above deviates from 1, that is the bug to fix in estimator.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
