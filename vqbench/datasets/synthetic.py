"""
Synthetic dataset generators.

All randomness flows from deterministic seeds — reproducibility guarantee (§10).
"""

from __future__ import annotations

import numpy as np


def random_unit_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """
    Generate n random unit vectors in R^d uniformly on the sphere.

    Method: sample from N(0, I_d), then normalize each row.

    Args:
        n: Number of vectors.
        d: Dimensionality.
        seed: Random seed.

    Returns:
        Array of shape (n, d) with each row having unit norm.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def controlled_ip_pairs(
    n: int,
    d: int,
    target_ip: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs (x, y) of unit vectors with E[⟨x, y⟩] ≈ target_ip.

    Method: y = target_ip * x + √(1 - target_ip²) * z, then normalize,
    where z is a random unit vector orthogonal to x.

    Args:
        n: Number of pairs.
        d: Dimensionality.
        target_ip: Desired expected inner product.
        seed: Random seed.

    Returns:
        (X, Y) each of shape (n, d).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    Z = rng.standard_normal((n, d))
    # Gram-Schmidt: remove component along x
    proj = np.sum(Z * X, axis=1, keepdims=True)
    Z = Z - proj * X
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    cos_a = target_ip
    sin_a = np.sqrt(max(1.0 - cos_a ** 2, 0.0))
    Y = cos_a * X + sin_a * Z
    # Y is already unit norm by construction, but normalize for safety
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    return X, Y
