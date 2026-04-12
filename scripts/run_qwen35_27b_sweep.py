"""
Qwen3.5-27B AWQ int4 deployment sweep — the Phase 9.2.2 headline run.

This is the **final headline run** for VQBench. Target: compress Qwen3.5-27B
KV cache on a 48 GB M5 Pro and measure ΔPPL against the fp16 AWQ baseline.

Memory budget (projected from Qwen3.5-4B scaling, see REPORT.md §7):
    Model weights (AWQ int4)   ~18 GB
    KV cache (128K ctx, Block B=64 K=V=4, 16 full-attn layers)   ~4.25 GB
    Linear attention state     ~50 MB
    Activations + PyTorch       ~2 GB
    ----------------------------------
    Total                      ~24.3 GB
    Headroom on 48 GB M5 Pro   ~23.7 GB

Estimated wall clock per config: 30–60 min on M5 Pro MPS. A full 3-bit and
4-bit sweep is ~4–8 hours. Results are written to --output-dir (previous
results are overwritten on each fresh run).

Requires the [validation] extras AND autoawq for the int4 weights:
    pip install -e '.[test,validation]'
    pip install autoawq

Usage:
    python scripts/run_qwen35_27b_sweep.py --output-dir results/qwen35_27b/
    python scripts/run_qwen35_27b_sweep.py --bits 4 --max-tokens 1024
    python scripts/run_qwen35_27b_sweep.py --dry-run  # print plan, don't load model

See PLAN.md §4.2 task 9.2.2. Before using this script, confirm that:
  1. The AWQ int4 checkpoint for Qwen3.5-27B is available on Hugging Face
     (update --model if the canonical name differs at runtime).
  2. autoawq is installed and imports cleanly.
  3. You have ~24 GB of free unified memory.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B-AWQ",
                        help="AWQ int4 checkpoint (update if the published name differs)")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4],
                        help="K cache bit-widths to sweep (default: 3 4)")
    parser.add_argument("--methods", nargs="+",
                        default=["TurboQuantMSE", "BlockTurboQuantMSE-B32", "BlockTurboQuantMSE-B64"],
                        help="K quantizer methods to sweep")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="tokens per eval slice (longer = stronger signal, slower)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="results/qwen35_27b/")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the sweep plan without loading the model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configs = [
        {"method": m, "bits": b}
        for m in args.methods for b in args.bits
    ]
    print(f"Sweep plan ({len(configs)} configs + 1 baseline):")
    print(f"  model:      {args.model}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  chunk:      {args.chunk}")
    print(f"  device:     {args.device}")
    print(f"  output:     {args.output_dir}")
    for c in configs:
        print(f"    * {c['method']} {c['bits']}-bit K+V")

    if args.dry_run:
        print("\n[dry-run] not loading model; exiting.")
        return 0

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "This script needs the [validation] extras. "
            "Run: pip install -e '.[test,validation]'"
        ) from e
    try:
        import awq  # noqa: F401
    except ImportError:
        print("WARNING: autoawq not found. If the checkpoint is AWQ int4, install it first: "
              "pip install autoawq")

    from vqbench.datasets.wikitext import load_wikitext2_encodings
    from vqbench.validation.monkey_patch import patch_model_kv, unpatch_model
    from vqbench.validation.streaming_ppl import evaluate_streaming_ppl

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "sweep_results.jsonl")

    print(f"\nLoading {args.model} (this may take several minutes) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True, device_map=args.device,
    ).eval()
    print(f"  loaded in {time.time() - t0:.1f} s")

    enc = load_wikitext2_encodings(tok, max_tokens=args.max_tokens)
    open(results_path, "w").close()  # truncate after model loads — no stale rows

    # ---- baseline -----
    print("\n=== fp16 baseline ===")
    t0 = time.time()
    ppl_base, n_scored = evaluate_streaming_ppl(
        model, enc, device=args.device, chunk=args.chunk, cache=None,
    )
    base_secs = time.time() - t0
    print(f"  PPL = {ppl_base:.4f}  ({n_scored} tokens scored, {base_secs:.1f} s)")

    with open(results_path, "a") as fh:
        fh.write(json.dumps({
            "config": "baseline_fp16",
            "ppl": ppl_base,
            "delta_ppl": 0.0,
            "delta_pct": 0.0,
            "n_scored": n_scored,
            "wall_secs": base_secs,
        }) + "\n")

    # ---- sweep -----
    def make_factory(method_name: str, bits: int):
        run_seed = args.seed
        def factory(d: int, seed: int = 0):
            actual_seed = run_seed * 1000 + seed
            if method_name == "TurboQuantMSE":
                from vqbench.methods.turboquant.mse import TurboQuantMSE
                return TurboQuantMSE(d=d, num_bits=bits, seed=actual_seed, norm_correction=True)
            if method_name.startswith("BlockTurboQuantMSE-B"):
                from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
                block_size = int(method_name.split("-B")[1])
                return BlockTurboQuantMSE(
                    d=d, num_bits=bits, block_size=block_size, seed=actual_seed, norm_correction=True,
                )
            raise ValueError(f"Unknown method: {method_name}")
        return factory

    for c in configs:
        tag = f"{c['method']} K{c['bits']}"
        print(f"\n=== {tag} ===")
        t0 = time.time()
        factory = make_factory(c["method"], c["bits"])
        hooks = patch_model_kv(model, k_quantizer_factory=factory, v_quantizer_factory=factory)
        try:
            ppl_q, _ = evaluate_streaming_ppl(
                model, enc, device=args.device, chunk=args.chunk, cache=None,
            )
        finally:
            unpatch_model(hooks)
        secs = time.time() - t0

        delta = ppl_q - ppl_base
        pct = (delta / ppl_base) * 100.0
        print(f"  PPL = {ppl_q:.4f}  ΔPPL = {delta:+.4f} ({pct:+.2f}%), {secs:.1f} s")

        with open(results_path, "a") as fh:
            fh.write(json.dumps({
                "config": tag,
                "method": c["method"],
                "bits_k": c["bits"],
                "ppl": ppl_q,
                "delta_ppl": delta,
                "delta_pct": pct,
                "wall_secs": secs,
            }) + "\n")

    print(f"\nDone. Results appended to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
