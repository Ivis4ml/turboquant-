"""
Quickstart 4/4 — BlockTurboQuantMSE end-to-end demo.

Reproduces the snippets from BlockTQ.md §9 (Quick use + KV cache integration).
Pure NumPy for the single-vector demo; the HF integration part is opt-in via
--with-hf-cache (requires [validation] extras) and is skipped by default.

BlockTurboQuantMSE matters at head_dim <= 128 (Qwen3, Llama 3 etc). At
head_dim = 256 (Qwen3.5, Gemma-4) scalar TurboQuantMSE is already lossless,
so block quantization gives no additional quality — see REPORT.md §3 and §4.

Usage:
    python scripts/quickstart_block_turboquant.py
    python scripts/quickstart_block_turboquant.py --d 128 --block-size 16 --bits 4
    python scripts/quickstart_block_turboquant.py --with-hf-cache \\
        --model Qwen/Qwen2.5-1.5B --block-size 32
"""

from __future__ import annotations

import argparse

import numpy as np

from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
from vqbench.methods.turboquant.mse import TurboQuantMSE


def demo_single_vector(d: int, block_size: int, bits: int, data_seed: int, quant_seed: int) -> None:
    print(f"Single-vector BlockTurboQuantMSE(d={d}, block_size={block_size}, bits={bits})")

    rng = np.random.default_rng(data_seed)
    x = rng.standard_normal(d)

    block_q = BlockTurboQuantMSE(d=d, num_bits=bits, block_size=block_size, seed=quant_seed)
    scalar_q = TurboQuantMSE(d=d, num_bits=bits, seed=quant_seed)

    qv_block = block_q.quantize(x)
    x_block = block_q.dequantize(qv_block)

    qv_scalar = scalar_q.quantize(x)
    x_scalar = scalar_q.dequantize(qv_scalar)

    def nmse(x: np.ndarray, x_hat: np.ndarray) -> float:
        return float(np.sum((x - x_hat) ** 2) / np.sum(x * x))

    print(f"  BlockTurboQuantMSE B={block_size}:  "
          f"{block_q.storage_bits(qv_block)} bits, nMSE={nmse(x, x_block):.6f}")
    print(f"  Scalar TurboQuantMSE:       "
          f"{scalar_q.storage_bits(qv_scalar)} bits, nMSE={nmse(x, x_scalar):.6f}")
    print(f"  Extra metadata cost:        "
          f"{block_q.storage_bits(qv_block) - scalar_q.storage_bits(qv_scalar)} bits "
          f"({(d // block_size)} blocks * 16-bit fp scale)")


def demo_hf_cache(model_id: str, block_size: int, bits: int, device: str) -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "--with-hf-cache needs the [validation] extras. "
            "Run: pip install -e '.[test,validation]'"
        ) from e

    from vqbench.torch_wrapper.hook import apply_quantized_cache

    print(f"\nHF cache integration: {model_id}, Block B={block_size} K, TQ-MSE V")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()

    method_key = f"BlockTurboQuantMSE-B{block_size}"
    cache = apply_quantized_cache(
        model,
        method_key=method_key,
        method_value="TurboQuantMSE",
        num_bits=bits,
    )
    prompt = "The key insight of TurboQuant is"
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30, past_key_values=cache)
    print(f"Prompt:    {prompt}")
    print(f"Generated: {tok.decode(out[0], skip_special_tokens=True)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=128, help="head_dim (default 128 — BlockTQ's target regime)")
    parser.add_argument("--block-size", type=int, default=16, help="block size: 16, 32, or 64")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=0, help="rng seed for synthetic vector")
    parser.add_argument("--quant-seed", type=int, default=42, help="Haar rotation seed (must differ from --data-seed)")
    parser.add_argument("--with-hf-cache", action="store_true",
                        help="also run the HF transformers Cache demo (needs [validation] extras)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                        help="model id for --with-hf-cache (must be full-attention only)")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if args.d % args.block_size != 0:
        raise SystemExit(f"d ({args.d}) must be divisible by block_size ({args.block_size})")

    if args.data_seed == args.quant_seed:
        print(f"WARNING: data_seed == quant_seed == {args.data_seed}. The data will align "
              f"with the rotation matrix and nMSE will be ~30x worse than normal.")

    demo_single_vector(args.d, args.block_size, args.bits, args.data_seed, args.quant_seed)

    if args.with_hf_cache:
        demo_hf_cache(args.model, args.block_size, args.bits, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
