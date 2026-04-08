"""
Quantize/dequantize/IP speed benchmarks.

Req N2: Single vector at d=512 in < 1ms
Req N3: Batch 1000 vectors at d=128 in < 50ms
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from vqbench.core.base import VectorQuantizer
from vqbench.datasets.synthetic import random_unit_vectors


@dataclass
class SpeedResult:
    method: str
    d: int
    num_bits: int
    quantize_single_ms: float
    dequantize_single_ms: float
    quantize_batch_ms: float  # 1000 vectors


def speed_benchmark(
    quantizer_classes: list[type[VectorQuantizer]],
    d: int = 512,
    num_bits: int = 2,
    n_batch: int = 1000,
    n_warmup: int = 10,
    n_repeat: int = 50,
    data_seed: int = 0,
    method_seed: int = 42,
) -> list[SpeedResult]:
    """Benchmark quantize/dequantize speed."""
    X = random_unit_vectors(n_batch, d, seed=data_seed)
    results = []

    for cls in quantizer_classes:
        try:
            q = cls(d=d, num_bits=num_bits, seed=method_seed)
        except (ValueError, TypeError):
            continue

        if hasattr(q, "fit"):
            q.fit(X)

        x = X[0]

        # Warmup
        for _ in range(n_warmup):
            qv = q.quantize(x)
            q.dequantize(qv)

        # Single-vector quantize
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            qv = q.quantize(x)
        t_quant = (time.perf_counter() - t0) / n_repeat * 1000

        # Single-vector dequantize
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            q.dequantize(qv)
        t_dequant = (time.perf_counter() - t0) / n_repeat * 1000

        # Batch quantize
        # Warmup
        q.quantize_batch(X[:10])
        t0 = time.perf_counter()
        q.quantize_batch(X)
        t_batch = (time.perf_counter() - t0) * 1000

        results.append(SpeedResult(
            method=q.name,
            d=d,
            num_bits=num_bits,
            quantize_single_ms=t_quant,
            dequantize_single_ms=t_dequant,
            quantize_batch_ms=t_batch,
        ))
    return results


def print_speed_table(results: list[SpeedResult]) -> None:
    print(f"{'Method':<18} {'d':>5} {'bits':>4} {'quant_1(ms)':>12} {'dequant_1(ms)':>14} {'quant_1k(ms)':>13}")
    print("-" * 72)
    for r in results:
        print(f"{r.method:<18} {r.d:>5} {r.num_bits:>4} {r.quantize_single_ms:>12.3f} "
              f"{r.dequantize_single_ms:>14.3f} {r.quantize_batch_ms:>13.1f}")
