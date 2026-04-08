"""
Shared random rotation module.

Paper refs:
  - TurboQuant (arXiv 2504.19874): Π ∈ R^{d×d} Haar-random orthogonal matrix
  - RaBitQ (arXiv 2405.12497): P ∈ R^{d×d} random orthogonal matrix
Both use the same Haar rotation — same (d, seed) → identical matrix.

Critical invariant: haar_rotation(d, seed) returns the same matrix regardless
of which method calls it. This ensures fairness (§10 of PLAN).
"""

from __future__ import annotations

import numpy as np


def haar_rotation(d: int, seed: int) -> np.ndarray:
    """
    Haar-random orthogonal matrix Π ∈ R^{d×d} via QR of Gaussian matrix.

    Paper: Stewart (1980), "The Efficient Generation of Random Orthogonal Matrices"
    Method: Q, R = QR(G) where G_{ij} ~ N(0,1), then Q ← Q @ diag(sign(diag(R)))
    to fix the sign ambiguity and obtain a proper Haar-distributed matrix.

    Args:
        d: Dimensionality.
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (d, d) with det = ±1.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    # Fix sign ambiguity for proper Haar distribution
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]  # Q @ diag(signs)
    return Q


def fast_walsh_hadamard(x: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform: O(d log d) structured rotation.

    Implements D·H·x where H is the normalized Hadamard matrix and D = diag(signs).
    Can be composed as D1·H·D2·H·D3 for a pseudo-random rotation.

    Paper: TurboQuant §4.2 — optional acceleration replacing dense Π.

    Args:
        x: Input vector of length d (must be power of 2).
        signs: Random ±1 signs for diagonal D. If None, no sign flip.

    Returns:
        Transformed vector of length d.
    """
    d = len(x)
    if d & (d - 1) != 0:
        raise ValueError(f"d must be a power of 2 for FWHT, got {d}")

    y = x.astype(np.float64).copy()

    if signs is not None:
        y *= signs

    # In-place butterfly
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                a = y[j]
                b = y[j + h]
                y[j] = a + b
                y[j + h] = a - b
        h *= 2

    # Normalize so that H is orthogonal: H/√d
    y /= np.sqrt(d)
    return y
