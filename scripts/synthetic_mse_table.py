"""
Synthetic MSE distortion table — reproduces REPORT.md §3 historical §3.1.

Runs every quantizer in VQBench on the same 2000 random unit vectors, reports
normalized reconstruction MSE = E[||x - x_hat||^2 / ||x||^2]. Pure NumPy.

Methods:
    TurboQuantMSE        b = 1..4  (Lloyd-Max on N(0, 1/d), norm_correction=True)
    TurboQuantProd       b = 2..4  (MSE at b-1 bits + QJL 1 bit)
    BlockTurboQuantMSE   b = 1..4  (configurable block_size, default d/2)
    RaBitQ 1-bit         b = 1     (hypercube, needs fit())
    ExtRaBitQ            b = 2..4  (bit-plane integer codebook, needs fit())

Theory (TurboQuant Theorem 1): normalized MSE for TQ-MSE ≤ (√(3π)/2) · 4^(-b).
For a Lloyd-Max optimized codebook the paper values are {0.36, 0.117, 0.03, 0.009}
at b = 1..4. The norm_correction=True variant (production setting, matches
turboquant_plus) is slightly higher at low b because it trades reconstruction
fidelity for improved IP bias.

Usage:
    python scripts/synthetic_mse_table.py                         # single seed
    python scripts/synthetic_mse_table.py --seeds 42 43 44        # variance
    python scripts/synthetic_mse_table.py --d 256 --seeds 42 43 44
"""

from __future__ import annotations

import argparse

import numpy as np

from vqbench.core.metrics import mse_distortion
from vqbench.datasets.synthetic import random_unit_vectors
from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.methods.turboquant.prod import TurboQuantProd


def _run(q, X: np.ndarray) -> float:
    if hasattr(q, "fit"):
        q.fit(X)
    qvs = q.quantize_batch(X)
    X_hat = q.dequantize_batch(qvs)
    return mse_distortion(X, X_hat)


def _cell(values: list[float]) -> str:
    if not values:
        return f"{'-':>16s}"
    if len(values) == 1:
        return f"{values[0]:>16.4f}"
    mean = float(np.mean(values))
    std = float(np.std(values))
    return f"{mean:>7.4f}±{std:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=256,
                        help="vector dimension (default 256, Qwen3.5 head_dim; 512 for old REPORT parity)")
    parser.add_argument("--n", type=int, default=2000, help="number of random unit vectors")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--block-size", type=int, default=None,
                        help="block size for BlockTurboQuantMSE (default d//2 if d % 2 == 0, else skipped)")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="rotation seeds (pass multiple for mean±std reporting)")
    args = parser.parse_args()

    for qs in args.seeds:
        if args.data_seed == qs:
            raise SystemExit(
                f"data_seed == quant_seed == {args.data_seed}. Set different values; "
                "otherwise the data aligns with the rotation matrix and MSE is ~30x worse."
            )

    block_size = args.block_size
    if block_size is None:
        block_size = args.d // 2 if args.d % 2 == 0 else None

    X = random_unit_vectors(args.n, args.d, seed=args.data_seed)

    print(f"\n=== Synthetic normalized MSE  (d = {args.d}, n = {args.n}) ===")
    print(f"    data_seed={args.data_seed}, seeds={args.seeds}, "
          f"{'mean ± std' if len(args.seeds) > 1 else 'single-seed'}")
    print(f"    Theory (Lloyd-Max bound 4^-b): "
          f"{{{', '.join(f'{4.0**-b:.4f}' for b in args.bits)}}}")
    print()

    methods: list[tuple[str, object]] = [
        ("TQ-MSE", lambda b, s: TurboQuantMSE(d=args.d, num_bits=b, seed=s, norm_correction=True)),
        ("TQ-Prod", lambda b, s: TurboQuantProd(d=args.d, num_bits=b, seed=s, norm_correction=True)),
    ]
    if block_size is not None:
        methods.append((f"Block B={block_size}",
                        lambda b, s, bs=block_size: BlockTurboQuantMSE(
                            d=args.d, num_bits=b, block_size=bs, seed=s, norm_correction=True)))
    methods.append(("RaBitQ 1-bit",
                    lambda b, s: RaBitQ1Bit(d=args.d, seed=s) if b == 1 else None))
    methods.append(("ExtRaBitQ",
                    lambda b, s: ExtRaBitQ(d=args.d, num_bits=b, seed=s) if b >= 2 else None))

    header = f"{'bits':>4s}"
    for label, _ in methods:
        header += f" {label:>16s}"
    header += f"  {'4^(-b)':>8s}"
    print(header)
    print("-" * len(header))

    for b in args.bits:
        row = f"{b:>4d}"
        for _label, factory in methods:
            values: list[float] = []
            for seed in args.seeds:
                try:
                    q = factory(b, seed)
                except (ValueError, TypeError):
                    q = None
                if q is None:
                    continue
                values.append(_run(q, X))
            row += f" {_cell(values):>16s}"
        row += f"  {4.0 ** -b:>8.4f}"
        print(row)

    print("\nNotes:")
    print("  * norm_correction=True throughout (matches turboquant_plus production).")
    print("  * Lower is better. Theory: MSE -> 4^(-b) as d -> infinity for Lloyd-Max.")
    print("  * TQ-Prod splits its budget (b-1 MSE bits + 1 QJL bit), so its")
    print("    reconstruction MSE is much higher than TQ-MSE at the same nominal b.")
    print("  * RaBitQ1Bit is 1-bit-only (hypercube). ExtRaBitQ is b >= 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
