"""
Shared random rotation module.

Paper refs:
  - TurboQuant (arXiv 2504.19874): Π ∈ R^{d×d} Haar-random orthogonal matrix
  - RaBitQ (arXiv 2405.12497): P ∈ R^{d×d} random orthogonal matrix
Both use the same Haar rotation — same (d, seed) → identical matrix.

Critical invariant: haar_rotation(d, seed) returns the same matrix regardless
of which method calls it. This ensures fairness (§10 of PLAN).

Provides two rotation backends:
  1. Dense Haar rotation: exact, O(d²) matmul — default for correctness benchmarks
  2. StructuredRotation (D₁·H·D₂·H·D₃): O(d log d) — for production speed
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Module-level cache — avoids recomputing identical rotation matrices (§7.4)
# Key: (d, seed), Value: ndarray (read-only, never mutate)
# ---------------------------------------------------------------------------
_rotation_cache: dict[tuple[int, int], np.ndarray] = {}


def haar_rotation(d: int, seed: int) -> np.ndarray:
    """
    Haar-random orthogonal matrix Π ∈ R^{d×d} via QR of Gaussian matrix.

    Paper: Stewart (1980), "The Efficient Generation of Random Orthogonal Matrices"
    Method: Q, R = QR(G) where G_{ij} ~ N(0,1), then Q ← Q @ diag(sign(diag(R)))
    to fix the sign ambiguity and obtain a proper Haar-distributed matrix.

    Results are cached: same (d, seed) returns the same object without recomputing.
    Callers must NOT mutate the returned matrix.

    Args:
        d: Dimensionality.
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (d, d) with det = ±1.
    """
    key = (d, seed)
    if key in _rotation_cache:
        return _rotation_cache[key]

    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]                     # Q @ diag(signs)
    Q.flags.writeable = False                        # prevent mutation
    _rotation_cache[key] = Q
    return Q


# ---------------------------------------------------------------------------
# Vectorized Fast Walsh-Hadamard Transform — O(d log d)
# ---------------------------------------------------------------------------

def _fwht_1d(x: np.ndarray) -> np.ndarray:
    """FWHT on a single vector. d must be power of 2."""
    d = len(x)
    y = x.astype(np.float64).copy()
    h = 1
    while h < d:
        y_view = y.reshape(-1, 2 * h)
        lo = y_view[:, :h].copy()
        hi = y_view[:, h:]
        y_view[:, :h] = lo + hi
        y_view[:, h:] = lo - hi
        h *= 2
    y /= np.sqrt(d)
    return y


def fwht_batch(Y: np.ndarray) -> np.ndarray:
    """
    Vectorized FWHT on rows of Y ∈ R^{n×d}. d must be power of 2.

    Uses reshape-based butterfly — no Python per-element loops.
    O(n · d · log d) total.
    """
    n, d = Y.shape
    out = Y.astype(np.float64).copy()
    h = 1
    while h < d:
        # Reshape to (n, d/(2h), 2, h) for butterfly pairs
        out_view = out.reshape(n, -1, 2, h)
        lo = out_view[:, :, 0, :].copy()
        hi = out_view[:, :, 1, :]
        out_view[:, :, 0, :] = lo + hi
        out_view[:, :, 1, :] = lo - hi
        h *= 2
    out /= np.sqrt(d)
    return out


def fast_walsh_hadamard(x: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform: O(d log d) structured rotation.

    Implements D·H·x where H is the normalized Hadamard matrix and D = diag(signs).

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
    return _fwht_1d(y) * np.sqrt(d)  # undo internal normalization, re-apply
    # Actually let's just call the optimized version directly:


# Override: use the vectorized path for single vectors too
def fast_walsh_hadamard(x: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:  # noqa: F811
    """
    Fast Walsh-Hadamard Transform: O(d log d) structured rotation.

    Implements D·H·x where H is the normalized Hadamard matrix and D = diag(signs).
    """
    d = len(x)
    if d & (d - 1) != 0:
        raise ValueError(f"d must be a power of 2 for FWHT, got {d}")
    y = x.astype(np.float64).copy()
    if signs is not None:
        y *= signs
    return _fwht_1d(y)


# ---------------------------------------------------------------------------
# StructuredRotation: D₁·H·D₂·H·D₃ — O(d log d) pseudo-random rotation
# ---------------------------------------------------------------------------

_structured_cache: dict[tuple[int, int], "StructuredRotation"] = {}


class StructuredRotation:
    """
    Pseudo-random rotation via 3-layer randomized Hadamard: D₁·H·D₂·H·D₃.

    Paper: TurboQuant §4.2; Ailon & Chazelle (2006) "Fast JL Transform"

    Each Dᵢ = diag(sᵢ) where sᵢ ∈ {-1,+1}^d is a random sign flip.
    H = normalized Walsh-Hadamard matrix (applied via butterfly in O(d log d)).
    Three rounds of D·H suffice for near-Haar distribution at d ≥ 128.

    Requires d = power of 2. Non-power-of-2 d is padded then truncated.
    """

    def __init__(self, d: int, seed: int) -> None:
        self.d = d
        self.d_padded = 1 << int(np.ceil(np.log2(max(d, 1))))
        rng = np.random.default_rng(seed)
        # 3 independent random sign vectors
        self.signs = [
            rng.choice(np.array([-1.0, 1.0]), size=self.d_padded)
            for _ in range(3)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply D₁·H·D₂·H·D₃·x in O(d log d). Input/output shape (d,)."""
        y = np.zeros(self.d_padded, dtype=np.float64)
        y[: self.d] = x
        for s in reversed(self.signs):        # D₃ first, then H, then D₂, etc.
            y *= s                            # ← Dᵢ
            y = _fwht_1d(y)                   # ← H (normalized)
        return y[: self.d]

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply rotation to each row of X ∈ R^{n×d}. Fully vectorized."""
        n = X.shape[0]
        Y = np.zeros((n, self.d_padded), dtype=np.float64)
        Y[:, : self.d] = X
        for s in reversed(self.signs):
            Y *= s[np.newaxis, :]             # ← broadcast Dᵢ
            Y = fwht_batch(Y)                 # ← vectorized H
        return Y[:, : self.d]

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse rotation: H·D₃ · H·D₂ · H·D₁ · y  (H is self-inverse)."""
        x = np.zeros(self.d_padded, dtype=np.float64)
        x[: self.d] = y
        for s in self.signs:                  # forward order for inverse
            x = _fwht_1d(x)                  # ← H
            x *= s                            # ← Dᵢ
        return x[: self.d]

    def inverse_batch(self, Y: np.ndarray) -> np.ndarray:
        """Inverse rotation, batch. Rows of Y ∈ R^{n×d}."""
        n = Y.shape[0]
        X = np.zeros((n, self.d_padded), dtype=np.float64)
        X[:, : self.d] = Y
        for s in self.signs:
            X = fwht_batch(X)
            X *= s[np.newaxis, :]
        return X[:, : self.d]


def get_structured_rotation(d: int, seed: int) -> StructuredRotation:
    """Cached factory for StructuredRotation instances."""
    key = (d, seed)
    if key not in _structured_cache:
        _structured_cache[key] = StructuredRotation(d, seed)
    return _structured_cache[key]
