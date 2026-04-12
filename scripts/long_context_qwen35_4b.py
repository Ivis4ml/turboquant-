"""
Long-context stress test for Qwen3.5-4B KV compression.

Qwen3.5-4B 4-bit K is near-lossless on 512 tokens (REPORT.md §3), but
KV cache compression's actual value is at long context. This script
runs the headline configs at increasing max_tokens and reports the ΔPPL
trend as cache boundaries accumulate.

At chunk = 256, the number of cache boundaries (moments where past K/V
is read from compressed storage rather than fp16) is max_tokens/chunk − 1:
    512  → 1 boundary   (2 chunks)
    1024 → 3 boundaries (4 chunks)
    2048 → 7 boundaries (8 chunks)
    4096 → 15 boundaries (16 chunks)

If 4-bit K compression is truly lossless, ΔPPL should stay at 0 as
boundaries grow. If quantization error accumulates, ΔPPL will trend up.

Configs tested (4 K-only monkey-patch runs, see CONFIGS list):
  - Scalar TurboQuantMSE 4-bit K (the §3 headline row)
  - Block B=64 4-bit K (Phase 9.1 block variant)
  - ExtRaBitQ 4-bit K
  - Scalar TurboQuantMSE 3-bit K (stress test at lower bit budget)

Per length × config runtime scales roughly as O(length²) because attention
scales with past-cache length. On M5 Pro with cached weights the default
config takes ~8 minutes for 3 lengths × 4 configs.

Usage:
    python scripts/long_context_qwen35_4b.py
    python scripts/long_context_qwen35_4b.py --lengths 512 1024 2048
    python scripts/long_context_qwen35_4b.py --lengths 4096 --chunk 512
    python scripts/long_context_qwen35_4b.py --patch-v  # K+V symmetric
"""

from __future__ import annotations

import argparse
import json
import os
import time


CONFIGS = [
    ("TQ-MSE K=4",     "TurboQuantMSE",      4, None),
    ("Block B=64 K=4", "BlockTurboQuantMSE", 4, 64),
    ("ExtRaBitQ K=4",  "ExtRaBitQ",          4, None),
    ("TQ-MSE K=3",     "TurboQuantMSE",      3, None),
]


def _make_factory(method: str, bits: int, block_size: int | None, run_seed: int):
    def factory(d: int, seed: int = 0):
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
        if method == "ExtRaBitQ":
            from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
            return ExtRaBitQ(d=d, num_bits=bits, seed=actual_seed)
        raise ValueError(f"Unknown method: {method}")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="results/long_context/")
    parser.add_argument("--patch-v", action="store_true",
                        help="also patch V (K+V symmetric, default: K-only)")
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
    log_path = os.path.join(args.output_dir, "long_context_runs.jsonl")

    print(f"Loading {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(args.device).eval()
    print(f"  loaded in {time.time() - t0:.1f} s")

    max_len = max(args.lengths)
    enc_full = load_wikitext2_encodings(tok, max_tokens=max_len)
    print(f"  encoded {enc_full.size(1)} tokens (will truncate per-length)")
    open(log_path, "w").close()  # truncate after model loads — no stale rows

    patch_v_tag = "K+V" if args.patch_v else "K-only"
    print(f"\nSweep: lengths = {args.lengths}, chunk = {args.chunk}, {patch_v_tag}")

    # Results table: rows = config, cols = length
    grid: dict[tuple[str, int], dict] = {}

    for length in args.lengths:
        enc = enc_full[:, :length]
        n_chunks = (length + args.chunk - 1) // args.chunk
        n_boundaries = max(0, n_chunks - 1)
        print(f"\n--- length = {length} ({n_chunks} chunks, {n_boundaries} boundaries) ---")

        # fp16 baseline for this length
        print("  fp16 baseline ...", flush=True)
        t0 = time.time()
        ppl_base, n_scored = evaluate_streaming_ppl(
            model, enc, device=args.device, chunk=args.chunk, cache=None,
        )
        base_secs = time.time() - t0
        print(f"    PPL = {ppl_base:.4f}  ({n_scored} tokens, {base_secs:.1f} s)")
        grid[("fp16", length)] = {"ppl": ppl_base, "delta_pct": 0.0, "secs": base_secs}
        with open(log_path, "a") as fh:
            fh.write(json.dumps({
                "length": length, "n_chunks": n_chunks, "n_boundaries": n_boundaries,
                "config": "fp16_baseline", "patch_v": False,
                "ppl": ppl_base, "delta_ppl": 0.0, "delta_pct": 0.0,
                "wall_secs": base_secs,
            }) + "\n")

        for label, method, bits, block in CONFIGS:
            print(f"  {label} ...", flush=True)
            t0 = time.time()
            factory = _make_factory(method, bits, block, args.seed)
            v_factory = factory if args.patch_v else None
            hooks = patch_model_kv(
                model, k_quantizer_factory=factory, v_quantizer_factory=v_factory,
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
            print(f"    PPL = {ppl_q:.4f}  Δ = {delta:+.4f} ({pct:+.2f}%), {secs:.1f} s")
            grid[(label, length)] = {"ppl": ppl_q, "delta_pct": pct, "secs": secs}
            with open(log_path, "a") as fh:
                fh.write(json.dumps({
                    "length": length, "n_chunks": n_chunks, "n_boundaries": n_boundaries,
                    "config": label, "method": method, "bits": bits,
                    "block_size": block, "patch_v": args.patch_v,
                    "ppl": ppl_q, "delta_ppl": delta, "delta_pct": pct,
                    "wall_secs": secs,
                }) + "\n")

    # --- summary table ---
    print(f"\n{'=' * 78}")
    print(f"Summary — Qwen3.5-4B long-context stress ({patch_v_tag})")
    print(f"{'=' * 78}\n")
    lengths = args.lengths
    header = f"{'config':<18s}"
    for length in lengths:
        header += f" {f'{length} tok':>14s}"
    print(header)
    print("-" * len(header))

    # fp16 row
    line = f"{'fp16 baseline':<18s}"
    for length in lengths:
        ppl = grid[("fp16", length)]["ppl"]
        line += f" {ppl:>10.4f}      "
    print(line)

    for label, *_ in CONFIGS:
        line = f"{label:<18s}"
        for length in lengths:
            entry = grid.get((label, length))
            if entry is None:
                line += f" {'-':>14s}"
            else:
                pct = entry["delta_pct"]
                line += f" {pct:>+13.2f}%"
        print(line)

    print(f"\nRaw JSONL: {log_path}")
    print("\nHow to read:")
    print("  * If ΔPPL stays flat as length grows → compression is lossless at long ctx")
    print("  * If ΔPPL trends up → quantization error accumulates with more cache boundaries")
    print("  * ±1% is inside single-seed noise (see scripts/variance_qwen35_4b.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
