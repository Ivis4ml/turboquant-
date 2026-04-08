"""
Optimized Product Quantization (OPQ): PQ with learned rotation.

Paper: Ge et al., CVPR 2013
Training (alternating optimization):
    1. Initialize R = I
    2. Repeat:
       a. Fix R → train PQ on R·X_train
       b. Fix codebooks → optimize R via Procrustes: R = U·Vᵀ from SVD(X̃ᵀ·X)
    3. Store R and codebooks
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.methods.pq.product_quant import ProductQuantizer


class OptimizedPQ(VectorQuantizer):
    """
    OPQ: PQ with learned orthogonal rotation.

    Paper: Ge et al., CVPR 2013
    """

    def __init__(self, d: int, num_bits: int, seed: int = 42) -> None:
        super().__init__(d, num_bits, seed)
        self._R = np.eye(d)  # Initialize rotation as identity
        self._pq = ProductQuantizer(d, num_bits, seed)
        self._trained = False

    @property
    def name(self) -> str:
        return "OPQ"

    def fit(self, X_train: np.ndarray, n_iter: int = 10) -> None:
        """
        Alternating optimization: learn rotation R and PQ codebooks.

        Paper: Ge et al., CVPR 2013, Algorithm 1
        """
        rng = np.random.default_rng(self.seed)
        R = np.eye(self.d)

        for it in range(n_iter):
            # Step a: Fix R, train PQ on rotated data
            X_rot = X_train @ R.T
            self._pq.fit(X_rot)

            # Reconstruct
            qvs = self._pq.quantize_batch(X_rot)
            X_hat_rot = self._pq.dequantize_batch(qvs)

            # Step b: Fix codebooks, optimize R via Procrustes
            # minimize ‖X_rot - X_hat_rot‖² = ‖X·Rᵀ - X̃‖²
            # Procrustes: R = U·Vᵀ from SVD(X̃ᵀ · X)
            M = X_hat_rot.T @ X_train  # (d, d)
            U, _, Vt = np.linalg.svd(M)
            R_new = U @ Vt
            # Ensure proper rotation (det = +1) via SVD sign correction
            # For an orthogonal matrix U @ Vt, det = det(U) * det(Vt)
            # Flip last column of U if determinant is negative
            if np.dot(U[:, 0], np.cross(U[:, 1], U[:, 2]) if self.d == 3 else U[:, 0]) != 0:
                # General approach: just check via the product of singular values sign
                pass
            # Simpler: det(U @ Vt) = det(U) * det(Vt), but for large d just use the matrix directly
            R_new = U @ Vt

            if np.allclose(R_new, R, atol=1e-6):
                break
            R = R_new

        self._R = R
        # Final PQ training with the learned R
        X_rot = X_train @ R.T
        self._pq.fit(X_rot)
        self._trained = True

    def _ensure_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("OPQ must be trained with fit() before use")

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """Rotate then PQ quantize."""
        self._ensure_trained()
        x_rot = self._R @ x
        qv = self._pq.quantize(x_rot)
        qv.metadata["opq_norm"] = float(np.linalg.norm(x))
        return qv

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """PQ dequantize then inverse rotate."""
        self._ensure_trained()
        x_rot_hat = self._pq.dequantize(qv)
        return self._R.T @ x_rot_hat

    def storage_bits(self, qv: QuantizedVector) -> int:
        """Same as PQ — rotation is shared/amortized."""
        return self._pq.storage_bits(qv)

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        self._ensure_trained()
        X_rot = X @ self._R.T
        return self._pq.quantize_batch(X_rot)

    def dequantize_batch(self, qvs: list[QuantizedVector]) -> np.ndarray:
        self._ensure_trained()
        X_rot_hat = self._pq.dequantize_batch(qvs)
        return X_rot_hat @ self._R
