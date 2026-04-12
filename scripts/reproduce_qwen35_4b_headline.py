"""
Reproduce the Qwen3.5-4B headline from REPORT.md §3.

Default config reproduces the top row: Scalar TurboQuantMSE 4-bit K on
Qwen3.5-4B over WikiText-2 first 512 tokens, chunk = 256. Expected result:

    fp16 baseline:  PPL ≈ 10.3866
    TQ-MSE 4-bit K: PPL ≈ 10.3866  (ΔPPL ≈ 0.0000, +0.0%)

Qwen3.5 is a hybrid-attention model (8 full-attention + 24 linear-attention
layers). VQBenchCache does not yet route hybrid-attention updates, so this
script uses the monkey-patch fallback (wraps k_proj / v_proj to inject
quantize→dequantize inside the forward pass). That over-estimates degradation
relative to the faithful semantics — so ΔPPL ≈ 0 under monkey-patch is a
pessimistic result, and a native hybrid VQBenchCache (Phase 9.2.1) would give
equally good or better numbers.

Requires the [validation] extras: `pip install -e '.[test,validation]'`.
First run downloads the model (~8 GB). Runtime on M5 Pro: ~60 s.

Usage:
    python scripts/reproduce_qwen35_4b_headline.py
    python scripts/reproduce_qwen35_4b_headline.py --bits 3  # 3-bit K sweep
    python scripts/reproduce_qwen35_4b_headline.py --method BlockTurboQuantMSE --block-size 64
    python scripts/reproduce_qwen35_4b_headline.py --max-tokens 1024 --chunk 256
    python scripts/reproduce_qwen35_4b_headline.py --model Qwen/Qwen3-4B  # head_dim=128 (Block B=16 helps)
"""

from __future__ import annotations

import argparse


def _make_factory(method: str, bits: int, run_seed: int, block_size: int):
    def factory(d: int, seed: int = 0):
        actual_seed = run_seed * 1000 + seed
        if method == "TurboQuantMSE":
            from vqbench.methods.turboquant.mse import TurboQuantMSE
            return TurboQuantMSE(d=d, num_bits=bits, seed=actual_seed, norm_correction=True)
        if method == "BlockTurboQuantMSE":
            from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
            return BlockTurboQuantMSE(
                d=d, num_bits=bits, block_size=block_size, seed=actual_seed, norm_correction=True,
            )
        if method == "ExtRaBitQ":
            from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
            return ExtRaBitQ(d=d, num_bits=bits, seed=actual_seed)
        raise ValueError(f"Unknown method: {method}")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B",
                        help="HF model id (default Qwen/Qwen3.5-4B)")
    parser.add_argument("--method", default="TurboQuantMSE",
                        choices=["TurboQuantMSE", "BlockTurboQuantMSE", "ExtRaBitQ"],
                        help="K quantizer method")
    parser.add_argument("--bits", type=int, default=4, help="bits per dim for K cache")
    parser.add_argument("--block-size", type=int, default=16,
                        help="block size for BlockTurboQuantMSE (ignored otherwise)")
    parser.add_argument("--patch-v", action="store_true",
                        help="also monkey-patch V (default: K only, V stays fp16 in monkey-patch)")
    parser.add_argument("--chunk", type=int, default=256,
                        help="streaming chunk size (256 → 1 cache boundary for 512 tokens)")
    parser.add_argument("--max-tokens", type=int, default=512, help="evaluation length")
    parser.add_argument("--device", default="mps", help="mps | cpu | cuda")
    parser.add_argument("--seed", type=int, default=42)
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

    print(f"Loading {args.model} (first run downloads the model) ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(args.device).eval()

    print(f"Loading WikiText-2 test split (first {args.max_tokens} tokens) ...")
    enc = load_wikitext2_encodings(tok, max_tokens=args.max_tokens)

    print("\n=== fp16 baseline ===")
    ppl_base, n_scored = evaluate_streaming_ppl(
        model, enc, device=args.device, chunk=args.chunk, cache=None,
    )
    print(f"  PPL = {ppl_base:.4f}  ({n_scored} tokens scored)")

    method_tag = args.method
    if args.method == "BlockTurboQuantMSE":
        method_tag = f"BlockTurboQuantMSE-B{args.block_size}"
    print(f"\n=== {method_tag} {args.bits}-bit K{' + V' if args.patch_v else ''} ===")

    factory = _make_factory(args.method, args.bits, args.seed, args.block_size)
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

    delta = ppl_q - ppl_base
    pct = (delta / ppl_base) * 100.0 if ppl_base > 0 else 0.0
    print(f"  PPL = {ppl_q:.4f}  ΔPPL = {delta:+.4f}  ({pct:+.2f}%)")

    print("\nHeadline table row:")
    print(f"  model={args.model}  method={method_tag}  K={args.bits}  "
          f"fp16={ppl_base:.4f}  quant={ppl_q:.4f}  ΔPPL={delta:+.4f}  ({pct:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
