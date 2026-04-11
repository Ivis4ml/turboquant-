"""
Quickstart 1/4 — Quantize and reconstruct a single vector with TurboQuantMSE.

Reproduces the "Quantize a single vector" block from README.md §Three-minute quick-start.
No validation extras required (pure NumPy path).

IMPORTANT — keep the data seed and the quantizer seed independent. If you
reuse the same seed for both, the data vector accidentally aligns with the
rotation matrix and quantization quality degrades by ~30×. This is why the
defaults are `--data-seed 0 --quant-seed 42`.

Usage:
    python scripts/quickstart_quantize_vector.py
    python scripts/quickstart_quantize_vector.py --d 128 --bits 4
    python scripts/quickstart_quantize_vector.py --n-vectors 200  # stable mean nMSE
"""

from __future__ import annotations

import argparse

import numpy as np

from vqbench.methods.turboquant.mse import TurboQuantMSE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=256, help="vector dimension (default 256, Qwen3.5 head_dim)")
    parser.add_argument("--bits", type=int, default=4, help="bits per dimension (1, 2, 3, 4)")
    parser.add_argument("--n-vectors", type=int, default=100,
                        help="number of independent vectors to average nMSE over (default 100)")
    parser.add_argument("--data-seed", type=int, default=0, help="rng seed for test vectors")
    parser.add_argument("--quant-seed", type=int, default=42, help="seed for the Haar rotation")
    args = parser.parse_args()

    if args.data_seed == args.quant_seed:
        print(f"WARNING: data_seed == quant_seed == {args.data_seed}. The data will align "
              f"with the rotation matrix and nMSE will be ~30x worse than normal. "
              f"Set --data-seed to a different value.")

    rng = np.random.default_rng(args.data_seed)
    q = TurboQuantMSE(d=args.d, num_bits=args.bits, seed=args.quant_seed)

    nmses = []
    bits_per_vector: int | None = None
    for _ in range(args.n_vectors):
        x = rng.standard_normal(args.d)
        qv = q.quantize(x)
        x_hat = q.dequantize(qv)
        nmses.append(float(np.sum((x - x_hat) ** 2) / np.sum(x * x)))
        if bits_per_vector is None:
            bits_per_vector = q.storage_bits(qv)

    assert bits_per_vector is not None
    fp16_bits = 16 * args.d
    ratio = fp16_bits / bits_per_vector
    mean_nmse = float(np.mean(nmses))
    std_nmse = float(np.std(nmses))

    print(f"TurboQuantMSE(d={args.d}, bits={args.bits}, seed={args.quant_seed})")
    print(f"  n_vectors:     {args.n_vectors}  (data_seed={args.data_seed})")
    print(f"  storage:       {bits_per_vector} bits/vector  (fp16 = {fp16_bits} bits)")
    print(f"  ratio:         {ratio:.2f}x vs fp16")
    print(f"  nMSE mean:     {mean_nmse:.6f}  (std {std_nmse:.6f})")
    print(f"  nMSE theory:   ~0.009 at b=4  (Lloyd-Max N(0,1/d))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
