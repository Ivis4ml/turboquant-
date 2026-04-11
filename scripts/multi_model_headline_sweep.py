"""
Multi-model head_dim scaling sweep — head_dim 64 / 128 / 256.

Runs the headline PPL comparison across three cached Qwen checkpoints that
cover the three head_dim regimes VQBench cares about:

    Qwen/Qwen2.5-0.5B   head_dim = 64    24 layers (100% full-attention)
    Qwen/Qwen3-4B       head_dim = 128   36 layers (100% full-attention)
    Qwen/Qwen3.5-4B     head_dim = 256   32 layers (25% full, 75% linear)

For each model, runs an fp16 baseline followed by a set of K-only monkey-patch
configurations. Produces the empirically verified "head_dim scaling story"
that is currently reported in REPORT.md §4 as a single table with some
historical numbers carried over from pre-2026-04-11 runs.

All configurations quantize only K (monkey-patch `k_proj`). V stays in fp16
so the reported ΔPPL isolates the K quantization error. For K+V symmetric
results see scripts/reproduce_qwen35_4b_headline.py --patch-v.

Runtime: ~8 minutes total on M5 Pro with cached weights.

Usage:
    python scripts/multi_model_headline_sweep.py
    python scripts/multi_model_headline_sweep.py --models Qwen/Qwen3-4B Qwen/Qwen3.5-4B
    python scripts/multi_model_headline_sweep.py --output-dir results/multi_model/
"""

from __future__ import annotations

import argparse
import json
import os
import time


# Default sweep: 4 K-only configs per model.
DEFAULT_METHODS = [
    ("TurboQuantMSE", 4, None),
    ("TurboQuantMSE", 3, None),
    ("BlockTurboQuantMSE", 4, "auto"),
    ("ExtRaBitQ", 4, None),
]


def _best_block_size(d: int) -> int:
    """Pick a reasonable default block size for a given head_dim.

    d <= 64   -> B = 16   (4 blocks)
    d == 128  -> B = 16   (8 blocks; REPORT.md §4 reported this as best on Qwen3-4B)
    d >= 256  -> B = 64   (4 blocks; on Qwen3.5-4B B=64 matches turboquant_plus turbo4 storage)
    """
    if d <= 64:
        return 16
    if d == 128:
        return 16
    return 64


def _make_factory(method: str, bits: int, block_size: int | None, run_seed: int):
    def factory(d: int, seed: int = 0):
        # `seed` is the per-head index from quant_dequant_tensor.
        # Offset by run_seed * 1000 so each variance run has its own rotation space.
        actual_seed = run_seed * 1000 + seed
        if method == "TurboQuantMSE":
            from vqbench.methods.turboquant.mse import TurboQuantMSE
            return TurboQuantMSE(d=d, num_bits=bits, seed=actual_seed, norm_correction=True)
        if method == "BlockTurboQuantMSE":
            from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
            bs = block_size if block_size not in (None, "auto") else _best_block_size(d)
            return BlockTurboQuantMSE(
                d=d, num_bits=bits, block_size=bs, seed=actual_seed, norm_correction=True,
            )
        if method == "ExtRaBitQ":
            from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
            return ExtRaBitQ(d=d, num_bits=bits, seed=actual_seed)
        raise ValueError(f"Unknown method: {method}")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3.5-4B",
    ])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="rotation seeds (pass multiple for mean±std reporting)")
    parser.add_argument("--output-dir", default="results/multi_model/")
    args = parser.parse_args()

    try:
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
    results_path = os.path.join(args.output_dir, "sweep.jsonl")

    # aggregated[(model, config_tag)] = list of ΔPPL% values, one per seed
    aggregated: dict[tuple[str, str], dict] = {}

    for model_id in args.models:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_id}")
        print(f"{'=' * 70}")

        print("Loading model ...")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True,
        ).to(args.device).eval()
        load_secs = time.time() - t0

        head_dim = getattr(model.config, "head_dim", None)
        if head_dim is None:
            head_dim = model.config.hidden_size // model.config.num_attention_heads
        n_layers = getattr(model.config, "num_hidden_layers", 0)
        print(f"  loaded in {load_secs:.1f} s, head_dim={head_dim}, layers={n_layers}")

        enc = load_wikitext2_encodings(tok, max_tokens=args.max_tokens)

        # fp16 baseline (seed-independent)
        print("  fp16 baseline ...", flush=True)
        t0 = time.time()
        ppl_base, n_scored = evaluate_streaming_ppl(
            model, enc, device=args.device, chunk=args.chunk, cache=None,
        )
        base_secs = time.time() - t0
        print(f"    PPL = {ppl_base:.4f}  ({n_scored} tokens, {base_secs:.1f} s)")

        baseline_key = (model_id, "fp16_baseline")
        aggregated[baseline_key] = {
            "head_dim": head_dim,
            "baseline_ppl": ppl_base,
            "deltas": [0.0],  # no variance for fp16
            "ppls": [ppl_base],
        }
        with open(results_path, "a") as fh:
            fh.write(json.dumps({
                "model": model_id, "head_dim": head_dim,
                "config": "fp16_baseline", "run_seed": None,
                "ppl": ppl_base, "delta_pct": 0.0, "wall_secs": base_secs,
            }) + "\n")

        # quant configs × seeds
        for method, bits, block in DEFAULT_METHODS:
            tag = method
            if method == "BlockTurboQuantMSE":
                bs = _best_block_size(head_dim)
                tag = f"BlockTurboQuantMSE-B{bs}"
            config_tag = f"{tag} K={bits}"
            key = (model_id, config_tag)
            aggregated.setdefault(key, {
                "head_dim": head_dim, "baseline_ppl": ppl_base,
                "deltas": [], "ppls": [],
            })

            for run_seed in args.seeds:
                print(f"  {config_tag} seed={run_seed} ...", flush=True)
                t0 = time.time()
                try:
                    factory = _make_factory(method, bits, block, run_seed)
                    hooks = patch_model_kv(model, k_quantizer_factory=factory)
                    try:
                        ppl_q, _ = evaluate_streaming_ppl(
                            model, enc, device=args.device, chunk=args.chunk, cache=None,
                        )
                    finally:
                        unpatch_model(hooks)
                    secs = time.time() - t0
                    delta = ppl_q - ppl_base
                    pct = (delta / ppl_base * 100.0) if ppl_base > 0 else 0.0
                    print(f"    PPL = {ppl_q:.4f}  Δ = {delta:+.4f} ({pct:+.2f}%), "
                          f"{secs:.1f} s")
                except Exception as e:  # noqa: BLE001
                    print(f"    FAILED: {e}")
                    ppl_q, pct, secs = float("nan"), float("nan"), 0.0

                aggregated[key]["ppls"].append(ppl_q)
                aggregated[key]["deltas"].append(pct)
                with open(results_path, "a") as fh:
                    fh.write(json.dumps({
                        "model": model_id, "head_dim": head_dim,
                        "config": config_tag, "method": tag, "bits_k": bits,
                        "run_seed": run_seed, "ppl": ppl_q, "delta_pct": pct,
                        "wall_secs": secs,
                    }) + "\n")

        del model
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass

    # ---- Summary table ----
    n_seeds = len(args.seeds)
    mode_tag = f"{n_seeds}-seed mean ± std" if n_seeds > 1 else "single-seed"
    print(f"\n\n{'=' * 78}")
    print(f"Summary — head_dim scaling ({mode_tag})")
    print(f"{'=' * 78}")
    print(f"\n{'model':<22s} {'head_dim':>9s} {'config':<32s} {'ΔPPL':>18s}")
    print("-" * 82)

    import numpy as np
    for (model_id, config_tag), data in aggregated.items():
        model_short = model_id.split("/")[-1]
        hd = data["head_dim"]
        deltas = data["deltas"]
        if config_tag == "fp16_baseline":
            display = f"baseline PPL={data['baseline_ppl']:.4f}"
        elif len(deltas) == 1:
            display = f"{deltas[0]:+.2f}%"
        else:
            m = float(np.mean(deltas))
            s = float(np.std(deltas))
            display = f"{m:+.2f}% ± {s:.2f}%"
        print(f"{model_short:<22s} {hd:>9d} {config_tag:<32s} {display:>18s}")

    print(f"\nResults written to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
