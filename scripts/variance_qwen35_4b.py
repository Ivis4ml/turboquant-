"""
Multi-seed variance study on the Qwen3.5-4B headline configs.

Runs each §3 headline config with N different rotation seeds and reports
mean ± std. Answers the single-seed ambiguity in REPORT.md §3 (the one
where Block B=32 K=3 K-only lands on 10.3866 while Block B=32 K=4 K-only
lands on 10.4682 — a monotonicity violation that can only be noise).

Design notes:

  * Model and encodings are loaded ONCE (loading dominates per-config time).
  * Each config is run N_SEEDS times. Every run uses a different *rotation*
    seed per head: `head_idx + run_seed * 1000`. The data stream is deterministic
    (WikiText-2 first N tokens), so run-to-run PPL variance comes purely from
    the Haar rotation and the downstream quantization.
  * K-only configs use `patch_model_kv(k=factory, v=None)`; K+V configs use
    `patch_model_kv(k=factory, v=factory)`.
  * fp16 baseline is measured once (no seed dependency).

Requires [validation] extras. Runtime ~8 min on M5 Pro for 7 configs × 3 seeds
on 512 WikiText-2 tokens.

Usage:
    python scripts/variance_qwen35_4b.py
    python scripts/variance_qwen35_4b.py --seeds 42 43 44 45 46
    python scripts/variance_qwen35_4b.py --max-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable


# Headline configs to sweep. Each is (label, method, bits, block_size, patch_v).
HEADLINE_CONFIGS = [
    # TurboQuantMSE family
    ("TQ-MSE K=4",         "TurboQuantMSE",      4, None, False),
    ("TQ-MSE K=V=4",       "TurboQuantMSE",      4, None, True),
    ("TQ-MSE K=3",         "TurboQuantMSE",      3, None, False),
    ("TQ-MSE K=V=3",       "TurboQuantMSE",      3, None, True),
    # BlockTurboQuantMSE family
    ("Block B=64 K=4",     "BlockTurboQuantMSE", 4, 64,   False),
    ("Block B=64 K=V=4",   "BlockTurboQuantMSE", 4, 64,   True),
    ("Block B=32 K=3",     "BlockTurboQuantMSE", 3, 32,   False),
    ("Block B=32 K=V=3",   "BlockTurboQuantMSE", 3, 32,   True),
    # RaBitQ family — K-cache as attention-native compression
    ("RaBitQ1Bit K=1",     "RaBitQ1Bit",         1, None, False),
    ("ExtRaBitQ K=4",      "ExtRaBitQ",          4, None, False),
    ("ExtRaBitQ K=V=4",    "ExtRaBitQ",          4, None, True),
    ("ExtRaBitQ K=3",      "ExtRaBitQ",          3, None, False),
    ("ExtRaBitQ K=V=3",    "ExtRaBitQ",          3, None, True),
]


def _make_factory(method: str, bits: int, block_size: int | None, run_seed: int) -> Callable:
    def factory(d: int, seed: int = 0):
        # `seed` here is the per-head index passed by quant_dequant_tensor.
        # Offset by run_seed * 1000 so each run has its own rotation space.
        actual_seed = run_seed * 1000 + seed
        if method == "TurboQuantMSE":
            from vqbench.methods.turboquant.mse import TurboQuantMSE
            return TurboQuantMSE(d=d, num_bits=bits, seed=actual_seed, norm_correction=True)
        if method == "BlockTurboQuantMSE":
            from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
            return BlockTurboQuantMSE(
                d=d, num_bits=bits, block_size=block_size,
                seed=actual_seed, norm_correction=True,
            )
        if method == "RaBitQ1Bit":
            from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
            return RaBitQ1Bit(d=d, seed=actual_seed)
        if method == "ExtRaBitQ":
            from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
            return ExtRaBitQ(d=d, num_bits=bits, seed=actual_seed)
        raise ValueError(f"Unknown method: {method}")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="results/variance/")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "This script needs the [validation] extras. "
            "Run: pip install -e '.[test,validation]'"
        ) from e

    from vqbench.datasets.wikitext import load_wikitext2_encodings
    from vqbench.validation.monkey_patch import patch_model_kv, unpatch_model
    from vqbench.validation.streaming_ppl import evaluate_streaming_ppl

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "variance_runs.jsonl")

    print(f"Loading {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(args.device).eval()
    print(f"  loaded in {time.time() - t0:.1f} s")

    enc = load_wikitext2_encodings(tok, max_tokens=args.max_tokens)

    # --- fp16 baseline (seed-independent) ---
    print("\nfp16 baseline ...")
    t0 = time.time()
    ppl_base, n_scored = evaluate_streaming_ppl(
        model, enc, device=args.device, chunk=args.chunk, cache=None,
    )
    print(f"  PPL = {ppl_base:.4f}  ({n_scored} tokens, {time.time() - t0:.1f} s)")

    # --- multi-seed sweep ---
    results: dict[str, list[float]] = {label: [] for label, *_ in HEADLINE_CONFIGS}

    for label, method, bits, block, patch_v in HEADLINE_CONFIGS:
        print(f"\n=== {label} ===")
        for run_seed in args.seeds:
            t0 = time.time()
            factory = _make_factory(method, bits, block, run_seed)
            v_factory = factory if patch_v else None
            hooks = patch_model_kv(
                model,
                k_quantizer_factory=factory,
                v_quantizer_factory=v_factory,
            )
            try:
                ppl_q, _ = evaluate_streaming_ppl(
                    model, enc, device=args.device, chunk=args.chunk, cache=None,
                )
            finally:
                unpatch_model(hooks)
            secs = time.time() - t0
            delta = ppl_q - ppl_base
            pct = (delta / ppl_base * 100.0) if ppl_base > 0 else 0.0
            print(f"  seed={run_seed}: PPL={ppl_q:.4f}  Δ={delta:+.4f} ({pct:+.2f}%), "
                  f"{secs:.1f} s")
            results[label].append(ppl_q)
            with open(log_path, "a") as fh:
                fh.write(json.dumps({
                    "config": label,
                    "method": method,
                    "bits": bits,
                    "block_size": block,
                    "patch_v": patch_v,
                    "run_seed": run_seed,
                    "ppl": ppl_q,
                    "delta_ppl": delta,
                    "delta_pct": pct,
                    "wall_secs": secs,
                }) + "\n")

    # --- summary table ---
    print(f"\n{'=' * 78}")
    print(f"Summary — Qwen3.5-4B, {args.max_tokens} tokens, "
          f"{len(args.seeds)} seeds per config")
    print(f"{'=' * 78}")
    print(f"\nfp16 baseline: {ppl_base:.4f}")
    print(f"\n{'config':<24s} {'mean ΔPPL%':>12s} "
          f"{'std':>9s} {'min':>9s} {'max':>9s} {'n':>3s}")
    print("-" * 78)
    for label, _method, _bits, _block, _pv in HEADLINE_CONFIGS:
        ppls = np.array(results[label])
        deltas = (ppls - ppl_base) / ppl_base * 100.0
        print(f"{label:<24s} "
              f"{np.mean(deltas):>+11.3f}% "
              f"{np.std(deltas):>9.3f} "
              f"{np.min(deltas):>+8.2f}% "
              f"{np.max(deltas):>+8.2f}% "
              f"{len(ppls):>3d}")

    print(f"\nRaw JSONL: {log_path}")
    print("\nInterpretation:")
    print("  * std < 0.1% → claim is robust at sub-1% granularity")
    print("  * std ≈ ΔPPL mean → claim is inside single-seed noise floor")
    print("  * std > ΔPPL mean → single-seed conclusions should be discarded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
