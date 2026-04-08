"""
MSE and IP distortion sweep across methods, dimensions, and bit-widths.

Req F3: d ∈ {128, 256, 512, 1536, 3072}, b ∈ {1, 2, 3, 4}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vqbench.core.base import VectorQuantizer
from vqbench.core.metrics import mse_distortion, ip_distortion
from vqbench.datasets.synthetic import random_unit_vectors


@dataclass
class DistortionResult:
    method: str
    d: int
    num_bits: int
    mse: float
    ip_dist: float
    n_vectors: int


def distortion_sweep(
    quantizer_classes: list[type[VectorQuantizer]],
    dims: list[int] = [128, 256, 512],
    bits: list[int] = [1, 2, 3, 4],
    n_vectors: int = 1000,
    data_seed: int = 0,
    method_seed: int = 42,
    fit_fn=None,
) -> list[DistortionResult]:
    """
    Run MSE & IP distortion sweep across methods, dims, bits.

    Args:
        quantizer_classes: List of VectorQuantizer subclasses.
        dims: Dimensions to test.
        bits: Bit-widths to test.
        n_vectors: Number of test vectors per (d, b) setting.
        data_seed: Seed for data generation.
        method_seed: Seed for quantizer initialization.
        fit_fn: Optional callback(quantizer, X_train) for data-dependent methods.

    Returns:
        List of DistortionResult for each (method, d, b) combination.
    """
    results = []
    for d in dims:
        X = random_unit_vectors(n_vectors, d, seed=data_seed)
        Y = random_unit_vectors(n_vectors, d, seed=data_seed + 1)

        for cls in quantizer_classes:
            for b in bits:
                try:
                    q = cls(d=d, num_bits=b, seed=method_seed)
                except (ValueError, TypeError):
                    continue

                if fit_fn is not None:
                    fit_fn(q, X)
                elif hasattr(q, "fit"):
                    q.fit(X)

                qvs = q.quantize_batch(X)
                X_hat = q.dequantize_batch(qvs)

                mse = mse_distortion(X, X_hat)
                ip_dist = ip_distortion(X, X_hat, Y)

                results.append(DistortionResult(
                    method=q.name,
                    d=d,
                    num_bits=b,
                    mse=mse,
                    ip_dist=ip_dist,
                    n_vectors=n_vectors,
                ))
    return results


def print_distortion_table(results: list[DistortionResult]) -> None:
    """Print results as a formatted table."""
    print(f"{'Method':<18} {'d':>5} {'bits':>4} {'MSE':>10} {'IP Dist':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r.method:<18} {r.d:>5} {r.num_bits:>4} {r.mse:>10.5f} {r.ip_dist:>12.6f}")
