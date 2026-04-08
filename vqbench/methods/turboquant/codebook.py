"""
Lloyd-Max codebook on the rotated coordinate distribution f_X.

Paper: Zandieh et al., arXiv 2504.19874, Lemma 1 + §3.1
Formula: f_X(t) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-t²)^{(d-3)/2},  t ∈ [-1,1]

For d ≥ 128 this is well-approximated by N(0, 1/d).
We use the Gaussian approximation for Lloyd-Max (matches paper).

Precomputed reference centroids (×√d):
  b=1: ±0.7979
  b=2: ±0.4528, ±1.5104
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _gaussian_lloyd_max(
    num_bits: int,
    sigma: float,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lloyd-Max optimal scalar quantizer for N(0, σ²).

    Paper: Lloyd (1982), "Least Squares Quantization in PCM"
    Applied to the rotated coordinate distribution N(0, 1/d) where σ = 1/√d.

    Args:
        num_bits: Number of bits → 2^num_bits centroids.
        sigma: Standard deviation of the Gaussian.
        max_iter: Maximum Lloyd iterations.
        tol: Convergence tolerance on centroid change.

    Returns:
        (centroids, boundaries): sorted centroids and decision boundaries.
        centroids has shape (2^num_bits,), boundaries has shape (2^num_bits - 1,).
    """
    k = 2 ** num_bits

    # Initialize centroids uniformly in [-3σ, 3σ]
    centroids = np.linspace(-3 * sigma, 3 * sigma, k)

    for _ in range(max_iter):
        # Decision boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids = conditional mean of N(0,σ²) in each partition
        new_centroids = np.empty(k)
        edges = np.concatenate([[-np.inf], boundaries, [np.inf]])

        for i in range(k):
            lo, hi = edges[i], edges[i + 1]
            # E[X | lo < X < hi] for X ~ N(0, σ²)
            # = σ · (φ(lo/σ) - φ(hi/σ)) / (Φ(hi/σ) - Φ(lo/σ))
            lo_s = lo / sigma
            hi_s = hi / sigma
            prob = norm.cdf(hi_s) - norm.cdf(lo_s)
            if prob < 1e-30:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = sigma * (norm.pdf(lo_s) - norm.pdf(hi_s)) / prob

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Recompute final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


def lloyd_max_codebook(
    num_bits: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lloyd-Max codebook for rotated coordinates at dimension d.

    Paper: arXiv 2504.19874, §3.1 — "optimal scalar quantizer for f_X"
    For d ≥ 128 we use the N(0, 1/d) approximation.

    Args:
        num_bits: Bits per coordinate.
        d: Vector dimension (determines σ = 1/√d).

    Returns:
        (centroids, boundaries): arrays of length 2^num_bits and 2^num_bits - 1.
    """
    sigma = 1.0 / np.sqrt(d)
    return _gaussian_lloyd_max(num_bits, sigma)


# --- Precomputed reference values (×√d, dimension-independent) ---
# Paper: arXiv 2504.19874, Table 1

CENTROIDS_SCALED_REF = {
    1: np.array([-0.7979, 0.7979]),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
}


def verify_codebook(num_bits: int, d: int, rtol: float = 0.01) -> bool:
    """Check that computed centroids match paper reference values (×√d)."""
    if num_bits not in CENTROIDS_SCALED_REF:
        return True  # No reference to check against
    centroids, _ = lloyd_max_codebook(num_bits, d)
    scaled = centroids * np.sqrt(d)
    ref = CENTROIDS_SCALED_REF[num_bits]
    return bool(np.allclose(scaled, ref, rtol=rtol))
