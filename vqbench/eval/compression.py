"""
True storage cost with metadata overhead.

Req N4: Metadata overhead < 5% for n >= 1000 vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vqbench.core.base import VectorQuantizer
from vqbench.datasets.synthetic import random_unit_vectors


@dataclass
class CompressionResult:
    method: str
    d: int
    num_bits: int
    bits_per_vector: int
    bits_per_dim: float
    compression_ratio: float  # vs fp32
    metadata_overhead_pct: float


def compression_analysis(
    quantizer_classes: list[type[VectorQuantizer]],
    d: int = 512,
    bits: list[int] = [1, 2, 3, 4],
    method_seed: int = 42,
) -> list[CompressionResult]:
    """Analyze storage cost per method."""
    x = np.random.RandomState(0).randn(d).astype(np.float64)
    x = x / np.linalg.norm(x)
    X_train = random_unit_vectors(100, d, seed=0)
    results = []

    fp32_bits = d * 32

    for cls in quantizer_classes:
        for b in bits:
            try:
                q = cls(d=d, num_bits=b, seed=method_seed)
            except (ValueError, TypeError):
                continue

            if hasattr(q, "fit"):
                q.fit(X_train)

            qv = q.quantize(x)
            total_bits = q.storage_bits(qv)
            data_bits = b * d
            metadata_bits = total_bits - data_bits
            overhead = metadata_bits / total_bits * 100 if total_bits > 0 else 0

            results.append(CompressionResult(
                method=q.name,
                d=d,
                num_bits=b,
                bits_per_vector=total_bits,
                bits_per_dim=total_bits / d,
                compression_ratio=fp32_bits / total_bits,
                metadata_overhead_pct=overhead,
            ))
    return results


def print_compression_table(results: list[CompressionResult]) -> None:
    print(f"{'Method':<18} {'d':>5} {'bits':>4} {'total':>7} {'b/dim':>6} {'ratio':>7} {'meta%':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r.method:<18} {r.d:>5} {r.num_bits:>4} {r.bits_per_vector:>7} "
              f"{r.bits_per_dim:>6.2f} {r.compression_ratio:>7.1f}x {r.metadata_overhead_pct:>5.1f}%")
