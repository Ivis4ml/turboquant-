"""
Product Quantization (PQ): k-means codebook per subspace.

Paper: Jegou, Douze, Schmid, TPAMI 2011
Pipeline:
    TRAIN: X_train → split into m subspaces of d/m dims → k-means per subspace
    QUANTIZE: x → split → nearest centroid per subspace → store indices
    DEQUANT: lookup centroids → concatenate

PQ is the only OFFLINE (data-dependent) method in VQBench.
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector


def _kmeans(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 0) -> np.ndarray:
    """
    Simple k-means for codebook training.

    Args:
        X: Data matrix (n, d_sub).
        k: Number of centroids.
        max_iter: Max iterations.
        seed: For centroid initialization.

    Returns:
        Centroids of shape (k, d_sub).
    """
    rng = np.random.default_rng(seed)
    n, d_sub = X.shape
    # k-means++ initialization (incremental distance tracking)
    first = rng.integers(n)
    centroids_init = [X[first]]
    min_dists = np.sum((X - X[first]) ** 2, axis=1)  # dist to first centroid
    for _ in range(1, k):
        probs = min_dists / (min_dists.sum() + 1e-30)
        chosen = rng.choice(n, p=probs)
        centroids_init.append(X[chosen])
        new_dists = np.sum((X - X[chosen]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)  # incremental update
    centroids = np.array(centroids_init)

    # Precompute X squared norms for fast distance computation
    X_sq = np.sum(X ** 2, axis=1)  # (n,)

    for _ in range(max_iter):
        # Assign via expanded distance: ||x-c||² = ||x||² - 2x·c + ||c||²
        C_sq = np.sum(centroids ** 2, axis=1)  # (k,)
        dists = X_sq[:, None] - 2 * (X @ centroids.T) + C_sq[None, :]  # (n, k)
        labels = np.argmin(dists, axis=1)
        # Update
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(new_centroids, centroids, atol=1e-8):
            break
        centroids = new_centroids
    return centroids


class ProductQuantizer(VectorQuantizer):
    """
    Standard Product Quantization.

    Paper: Jegou et al., TPAMI 2011

    Bit budget: total bits = b·d. With m subspaces: each gets b·d/m bits → k = 2^(b·d/m).
    Default: m = d/4 (4-dim subspaces).
    """

    def __init__(self, d: int, num_bits: int, seed: int = 42) -> None:
        super().__init__(d, num_bits, seed)
        # m subspaces, each of dimension d_sub
        self._d_sub = 4  # 4-dim subspaces
        if d % self._d_sub != 0:
            raise ValueError(f"d={d} must be divisible by d_sub={self._d_sub}")
        self._m = d // self._d_sub
        # bits per subspace
        bits_per_sub = num_bits * self._d_sub  # b bits per dim × d_sub dims
        self._k = min(2 ** bits_per_sub, 256)  # cap at 256 for memory
        self._bits_per_sub = int(np.ceil(np.log2(self._k)))
        self._codebooks: list[np.ndarray] | None = None  # (m,) list of (k, d_sub) arrays
        self._trained = False

    @property
    def name(self) -> str:
        return "PQ"

    def fit(self, X_train: np.ndarray) -> None:
        """
        Train PQ codebooks via k-means per subspace.

        Args:
            X_train: Training data, shape (n_train, d).
        """
        self._codebooks = []
        for i in range(self._m):
            sub = X_train[:, i * self._d_sub:(i + 1) * self._d_sub]
            centroids = _kmeans(sub, self._k, seed=self.seed + i)
            self._codebooks.append(centroids)
        self._trained = True

    def _ensure_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("PQ must be trained with fit() before use")

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """Quantize by nearest centroid per subspace."""
        self._ensure_trained()
        x_norm = np.linalg.norm(x)
        indices = np.empty(self._m, dtype=np.int16)
        for i in range(self._m):
            sub = x[i * self._d_sub:(i + 1) * self._d_sub]
            dists = np.sum((self._codebooks[i] - sub) ** 2, axis=1)
            indices[i] = np.argmin(dists)
        return QuantizedVector(
            indices=indices,
            norms=np.array([x_norm], dtype=np.float32),
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """Concatenate centroids from each subspace."""
        self._ensure_trained()
        parts = []
        for i in range(self._m):
            parts.append(self._codebooks[i][qv.indices[i]])
        return np.concatenate(parts)

    def storage_bits(self, qv: QuantizedVector) -> int:
        """m × bits_per_sub + 16 bits (norm)."""
        return self._m * self._bits_per_sub + 16

    def ip_adc(self, query: np.ndarray, qv: QuantizedVector) -> float:
        """
        Asymmetric Distance Computation for fast IP.

        Paper: Jegou et al., TPAMI 2011, §5.2
        Precompute lookup table per subspace, then just look up and sum.
        """
        self._ensure_trained()
        total = 0.0
        for i in range(self._m):
            q_sub = query[i * self._d_sub:(i + 1) * self._d_sub]
            lut = self._codebooks[i] @ q_sub  # (k,)
            total += lut[qv.indices[i]]
        return float(total)

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Vectorized batch quantization."""
        self._ensure_trained()
        n = len(X)
        norms = np.linalg.norm(X, axis=1)
        all_indices = np.empty((n, self._m), dtype=np.int16)

        for i in range(self._m):
            sub = X[:, i * self._d_sub:(i + 1) * self._d_sub]  # (n, d_sub)
            # Vectorized distance computation
            dists = (
                np.sum(sub ** 2, axis=1, keepdims=True)
                - 2 * sub @ self._codebooks[i].T
                + np.sum(self._codebooks[i] ** 2, axis=1)
            )  # (n, k)
            all_indices[:, i] = np.argmin(dists, axis=1)

        return [
            QuantizedVector(
                indices=all_indices[j],
                norms=np.array([norms[j]], dtype=np.float32),
            )
            for j in range(n)
        ]
