"""
Quickstart 2/4 — Compress a KV cache with asymmetric K / V strategy.

Reproduces the "Compress a KV cache" block from README.md §Three-minute quick-start.
No validation extras required (pure NumPy path).

Recommended by head_dim (see REPORT.md §3 and §4):

    head_dim >= 256 (Qwen3.5, Gemma-4): scalar TurboQuantMSE for K and V
    head_dim == 128 (Qwen3, Llama 3):   BlockTurboQuantMSE-B16 for K, scalar for V
    head_dim <= 64  (Qwen2.5-0.5B):     out of the theory's intended regime; expect degradation

Usage:
    python scripts/quickstart_kv_cache.py
    python scripts/quickstart_kv_cache.py --head-dim 128 --seq-len 1024 --bits 3
"""

from __future__ import annotations

import argparse

import numpy as np

from vqbench.kv_cache.compressor import KVCacheCompressor
from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
from vqbench.methods.turboquant.mse import TurboQuantMSE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dim", type=int, default=256, help="head_dim (256 for Qwen3.5, 128 for Qwen3)")
    parser.add_argument("--seq-len", type=int, default=512, help="number of KV vectors to compress")
    parser.add_argument("--bits", type=int, default=4, help="bits per dimension")
    parser.add_argument("--block-size", type=int, default=None,
                        help="if set, K uses BlockTurboQuantMSE with this block size (forces Block K)")
    parser.add_argument("--data-seed", type=int, default=0, help="rng seed for synthetic K/V tensors")
    parser.add_argument("--quant-seed", type=int, default=42,
                        help="Haar rotation seed (must differ from --data-seed)")
    args = parser.parse_args()

    if args.data_seed == args.quant_seed:
        print(f"WARNING: data_seed == quant_seed == {args.data_seed}. The data will align "
              f"with the rotation matrix and nMSE will be ~30x worse than normal. "
              f"Set --data-seed to a different value.")

    rng = np.random.default_rng(args.data_seed)

    if args.block_size is not None:
        key_q = BlockTurboQuantMSE(
            d=args.head_dim, num_bits=args.bits, block_size=args.block_size, seed=args.quant_seed,
        )
        k_label = f"BlockTurboQuantMSE-B{args.block_size}"
    elif args.head_dim <= 128:
        key_q = BlockTurboQuantMSE(
            d=args.head_dim, num_bits=args.bits, block_size=16, seed=args.quant_seed,
        )
        k_label = "BlockTurboQuantMSE-B16 (auto for head_dim<=128)"
    else:
        key_q = TurboQuantMSE(d=args.head_dim, num_bits=args.bits, seed=args.quant_seed)
        k_label = "TurboQuantMSE (auto for head_dim>=256)"

    val_q = TurboQuantMSE(d=args.head_dim, num_bits=args.bits, seed=args.quant_seed + 1)

    comp = KVCacheCompressor(key_q, val_q)

    keys = rng.standard_normal((args.seq_len, args.head_dim))
    values = rng.standard_normal((args.seq_len, args.head_dim))
    comp.compress(keys, values)

    k_hat = comp.get_keys()
    v_hat = comp.get_values()

    k_nmse = float(np.mean(np.sum((keys - k_hat) ** 2, axis=1) / np.sum(keys * keys, axis=1)))
    v_nmse = float(np.mean(np.sum((values - v_hat) ** 2, axis=1) / np.sum(values * values, axis=1)))

    print(f"KV cache compression  (seq_len={args.seq_len}, head_dim={args.head_dim}, bits={args.bits})")
    print(f"  K quantizer:   {k_label}")
    print(f"  V quantizer:   TurboQuantMSE")
    print(f"  compression:   {comp.compression_ratio():.2f}x vs fp16")
    print(f"  K normalized MSE: {k_nmse:.6f}")
    print(f"  V normalized MSE: {v_nmse:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
