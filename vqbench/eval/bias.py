"""
IP bias analysis: fit multiplicative factor α for each method/bit-width.

Req F4: Fit α where E[⟨y, x̃⟩] ≈ α · ⟨y, x⟩
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vqbench.core.base import VectorQuantizer
from vqbench.core.metrics import ip_bias
from vqbench.datasets.synthetic import random_unit_vectors


@dataclass
class BiasResult:
    method: str
    num_bits: int
    alpha: float
    residual: float


def bias_sweep(
    quantizer_classes: list[type[VectorQuantizer]],
    d: int = 512,
    bits: list[int] = [1, 2, 3, 4],
    n_vectors: int = 2000,
    data_seed: int = 0,
    method_seed: int = 42,
) -> list[BiasResult]:
    """Fit α for each (method, bit-width)."""
    X = random_unit_vectors(n_vectors, d, seed=data_seed)
    Y = random_unit_vectors(n_vectors, d, seed=data_seed + 1)
    results = []

    for cls in quantizer_classes:
        for b in bits:
            try:
                q = cls(d=d, num_bits=b, seed=method_seed)
            except (ValueError, TypeError):
                continue

            if hasattr(q, "fit"):
                q.fit(X)

            qvs = q.quantize_batch(X)
            X_hat = q.dequantize_batch(qvs)
            alpha, residual = ip_bias(X, X_hat, Y)
            results.append(BiasResult(q.name, b, alpha, residual))
    return results


def print_bias_table(results: list[BiasResult]) -> None:
    print(f"{'Method':<18} {'bits':>4} {'α':>8} {'residual':>10}")
    print("-" * 45)
    for r in results:
        unbiased = "✓" if abs(r.alpha - 1.0) < 0.02 else ""
        print(f"{r.method:<18} {r.num_bits:>4} {r.alpha:>8.4f} {r.residual:>10.6f} {unbiased}")
