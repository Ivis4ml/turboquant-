"""
Quickstart 3/4 — Drop VQBenchCache into a HuggingFace transformers model.

Reproduces the "Drop into HuggingFace transformers" block from README.md
§Three-minute quick-start.

Requires the `[validation]` extras (torch + transformers + datasets). Downloads
the model the first time it runs (~3 GB for the default Qwen3-4B).

Current integration is single-batch (batch=1) and only exercises models with
pure full-attention. For hybrid-attention models (Qwen3.5 series) use the
monkey-patch fallback in scripts/reproduce_qwen35_4b_headline.py until Phase
9.2.1 lands the native hybrid VQBenchCache.

Usage:
    python scripts/quickstart_transformers_cache.py
    python scripts/quickstart_transformers_cache.py \\
        --model Qwen/Qwen2.5-1.5B --method-key BlockTurboQuantMSE-B16 --bits 4
    python scripts/quickstart_transformers_cache.py \\
        --prompt "The Turing test is" --max-new-tokens 40
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                        help="HF model id (default Qwen/Qwen2.5-1.5B; full-attention only)")
    parser.add_argument("--method-key", default="TurboQuantMSE",
                        help="K quantizer: TurboQuantMSE, BlockTurboQuantMSE-B{16,32,64}, ExtRaBitQ, ...")
    parser.add_argument("--method-value", default="TurboQuantMSE")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", default="mps", help="mps | cpu | cuda")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "This script needs the [validation] extras. Run: pip install -e '.[test,validation]'"
        ) from e

    from vqbench.torch_wrapper.hook import apply_quantized_cache

    print(f"Loading {args.model} (first run downloads the model) ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(args.device).eval()

    print(f"Attaching VQBenchCache: K={args.method_key}, V={args.method_value}, bits={args.bits}")
    cache = apply_quantized_cache(
        model,
        method_key=args.method_key,
        method_value=args.method_value,
        num_bits=args.bits,
    )

    inputs = tok(args.prompt, return_tensors="pt").to(args.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, past_key_values=cache)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"Prompt:    {args.prompt}")
    print(f"Generated: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
