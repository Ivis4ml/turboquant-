"""
ANN recall@k benchmark.

Req F5: recall@k on synthetic and GloVe data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vqbench.core.base import VectorQuantizer
from vqbench.core.metrics import recall_at_k
from vqbench.datasets.synthetic import random_unit_vectors


@dataclass
class RecallResult:
    method: str
    num_bits: int
    k: int
    recall: float


def recall_sweep(
    quantizer_classes: list[type[VectorQuantizer]],
    d: int = 128,
    bits: list[int] = [2, 3, 4],
    n_db: int = 5000,
    n_queries: int = 100,
    k: int = 10,
    data_seed: int = 0,
    method_seed: int = 42,
) -> list[RecallResult]:
    """Run recall@k sweep across methods and bit-widths."""
    X_db = random_unit_vectors(n_db, d, seed=data_seed)
    queries = random_unit_vectors(n_queries, d, seed=data_seed + 1)
    results = []

    for cls in quantizer_classes:
        for b in bits:
            try:
                q = cls(d=d, num_bits=b, seed=method_seed)
            except (ValueError, TypeError):
                continue

            if hasattr(q, "fit"):
                q.fit(X_db)

            X_hat = q.dequantize_batch(q.quantize_batch(X_db))
            recall = recall_at_k(X_db, X_hat, queries, k=k)
            results.append(RecallResult(q.name, b, k, recall))
    return results
