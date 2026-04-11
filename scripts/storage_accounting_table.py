"""
Storage accounting table — reproduces REPORT.md §6 per-vector storage.

Reports, for each method/bit-width combination at a given head_dim:
    data bits + metadata bits = total bits
    effective bits/dim
    compression ratio vs fp16  (what modern LLM KV caches actually use)

The KV-cache use case uses fp16 as the baseline. VQBench's generic
`eval/compression.py` helper reports vs fp32 because it was written for
the generic ANN / IP search benchmark; this script ignores that helper and
computes the fp16-relative figures directly so the numbers match REPORT §6.

Pure NumPy — no model download required.

Usage:
    python scripts/storage_accounting_table.py
    python scripts/storage_accounting_table.py --d 256 --bits 2 3 4
    python scripts/storage_accounting_table.py --d 128 --no-block
"""

from __future__ import annotations

import argparse

import numpy as np

from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.methods.turboquant.prod import TurboQuantProd


def _probe(q, d: int) -> int:
    """Quantize a dummy vector and return `storage_bits(qv)`."""
    x = np.random.default_rng(0).standard_normal(d)
    if hasattr(q, "fit"):
        q.fit(x.reshape(1, -1))
    return q.storage_bits(q.quantize(x))


def _row(label: str, total: int, data: int, d: int) -> None:
    meta = total - data
    eff = total / d
    vs_fp16 = (16 * d) / total
    print(f"{label:<28s}{data:>6d} {meta:>5d} {total:>7d} "
          f"{eff:>9.3f} {vs_fp16:>9.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=128,
                        help="head_dim (default 128; 256 for Qwen3.5)")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--no-block", action="store_true",
                        help="skip BlockTurboQuantMSE rows")
    args = parser.parse_args()

    fp16_baseline = 16 * args.d
    print(f"\n=== head_dim = {args.d}, fp16 baseline = {fp16_baseline} bits/vector ===\n")
    print(f"{'method':<28s}{'data':>6s} {'meta':>5s} {'total':>7s} "
          f"{'bits/dim':>9s} {'vs fp16':>10s}")
    print("-" * 70)

    for b in args.bits:
        tq = TurboQuantMSE(d=args.d, num_bits=b, seed=42)
        _row(f"TurboQuantMSE {b}-bit", _probe(tq, args.d), b * args.d, args.d)

    for b in args.bits:
        if b == 1:
            continue  # TQ-Prod minimum useful is 2-bit
        tp = TurboQuantProd(d=args.d, num_bits=b, seed=42)
        _row(f"TurboQuantProd {b}-bit", _probe(tp, args.d), b * args.d, args.d)

    if not args.no_block:
        for b in args.bits:
            for block_size in [16, 32, 64]:
                if args.d % block_size != 0:
                    continue
                bq = BlockTurboQuantMSE(d=args.d, num_bits=b, block_size=block_size, seed=42)
                _row(f"BlockTurboQuantMSE-B{block_size} {b}-bit",
                     _probe(bq, args.d), b * args.d, args.d)

    # RaBitQ 1-bit (only 1-bit)
    rq1 = RaBitQ1Bit(d=args.d, seed=42)
    _row("RaBitQ 1-bit", _probe(rq1, args.d), 1 * args.d, args.d)

    # ExtRaBitQ multi-bit
    for b in args.bits:
        if b < 2:
            continue
        erq = ExtRaBitQ(d=args.d, num_bits=b, seed=42)
        _row(f"ExtRaBitQ {b}-bit", _probe(erq, args.d), b * args.d, args.d)

    print("\nLegend:")
    print("  data    = num_bits * d")
    print("  meta    = per-vector metadata (norms, scales, ip_coeff, ...)")
    print("  total   = data + meta  (what storage_bits(qv) returns)")
    print("  bits/dim = total / d  (effective bit-width, includes metadata)")
    print("  vs fp16 = (16 * d) / total  (compression ratio vs fp16 KV cache)")
    print()
    print("Shared per-instance state (rotation matrix, QJL projection, RaBitQ")
    print("centroid, PQ codebooks) is NOT included. For n >= 10k vectors it's")
    print("negligible; for very small caches it dominates. See REPORT.md §6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
