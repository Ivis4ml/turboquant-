"""
TurboQuantProd — Algorithm 2: Unbiased IP via MSE + QJL residual correction.

Paper: Zandieh et al., arXiv 2504.19874, Algorithm 2
Pipeline:
    x → MSE quantize at (b-1) bits → x̃_mse
      → residual r = x − x̃_mse, store γ = ‖r‖
      → QJL on r: qjl = sign(S · r)
      → store (idx, qjl, γ)
    DeQuant: x̃ = x̃_mse + √(π/2)/d · γ · Sᵀ · qjl

Unbiasedness (Theorem 2):
    E[⟨y, x̃⟩] = ⟨y, x̃_mse⟩ + E[⟨y, r̃_qjl⟩] = ⟨y, x̃_mse + r⟩ = ⟨y, x⟩  ✓

IP distortion (Theorem 2):
    D_prod ≤ (π/2d) · ‖y‖² · D_mse(b-1)

Special case b=1: MSE stage = 0 bits (x̃_mse = 0), entire budget → QJL on x itself.
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.core.rotation import haar_rotation
from vqbench.methods.turboquant.codebook import lloyd_max_codebook
from vqbench.methods.turboquant.qjl import QJL_CONST


class TurboQuantProd(VectorQuantizer):
    """
    TurboQuant Algorithm 2: (b-1)-bit MSE + 1-bit QJL for unbiased IP.

    Paper: Zandieh et al., arXiv 2504.19874, Algorithm 2
    """

    def __init__(self, d: int, num_bits: int, seed: int = 42,
                 norm_correction: bool = True) -> None:
        super().__init__(d, num_bits, seed)
        self._rotation = haar_rotation(d, seed)
        self._norm_correction = norm_correction

        # MSE stage uses (b-1) bits; b=1 → 0 bits (pure QJL)
        self._mse_bits = max(num_bits - 1, 0)
        if self._mse_bits > 0:
            self._centroids, self._boundaries = lloyd_max_codebook(self._mse_bits, d)
        else:
            self._centroids = None
            self._boundaries = None

        # QJL projection matrix S — independent from rotation Π
        rng = np.random.default_rng(seed + 2**31)
        self._S = rng.standard_normal((d, d))

    @property
    def name(self) -> str:
        return "TurboQuantProd"

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """
        Algorithm 2 quantization.

        Paper: arXiv 2504.19874, Algorithm 2
        Step 1: MSE quantize at (b-1) bits
        Step 2: QJL on residual r = x - x̃_mse
        """
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-30:
            return QuantizedVector(
                indices=np.zeros(self.d, dtype=np.int8),
                norms=np.array([0.0, 0.0], dtype=np.float32),
                signs=np.ones(self.d, dtype=np.int8),
            )

        # --- MSE stage ---
        if self._mse_bits > 0:
            x_hat = x / x_norm
            y = self._rotation @ x_hat              # ← rotate
            indices = np.searchsorted(self._boundaries, y).astype(np.int8)
            y_hat = self._centroids[indices]         # ← MSE reconstruction in rotated space
            if self._norm_correction:
                y_norm = np.linalg.norm(y_hat)
                if y_norm > 1e-30:
                    y_hat = y_hat / y_norm
            x_mse = x_norm * (self._rotation.T @ y_hat)  # ← back to original space
        else:
            # b=1: no MSE stage, x̃_mse = 0           ← §5.5 special case
            indices = np.zeros(self.d, dtype=np.int8)
            x_mse = np.zeros(self.d)

        # --- QJL stage on residual ---
        residual = x - x_mse                        # ← r = x − x̃_mse
        gamma = np.linalg.norm(residual)             # ← γ = ‖r‖

        if gamma < 1e-30:
            signs = np.ones(self.d, dtype=np.int8)
        else:
            proj = self._S @ residual                # ← S · r
            signs = np.sign(proj).astype(np.int8)
            signs[signs == 0] = 1

        return QuantizedVector(
            indices=indices,
            norms=np.array([x_norm, gamma], dtype=np.float32),
            signs=signs,
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """
        Algorithm 2 dequantization.

        Paper: arXiv 2504.19874, Algorithm 2
        x̃ = x̃_mse + √(π/2)/d · γ · Sᵀ · qjl
        """
        x_norm = float(qv.norms[0])
        gamma = float(qv.norms[1])

        # MSE reconstruction
        if self._mse_bits > 0 and x_norm > 1e-30:
            y_hat = self._centroids[qv.indices]
            if self._norm_correction:                # ← turboquant_plus parity
                y_norm = np.linalg.norm(y_hat)
                if y_norm > 1e-30:
                    y_hat = y_hat / y_norm
            x_mse = x_norm * (self._rotation.T @ y_hat)
        else:
            x_mse = np.zeros(self.d)

        # QJL residual correction                    ← Definition 1
        if gamma > 1e-30:
            r_hat = QJL_CONST / self.d * gamma * (self._S.T @ qv.signs.astype(np.float64))
        else:
            r_hat = np.zeros(self.d)

        return x_mse + r_hat                         # ← x̃ = x̃_mse + r̃

    def storage_bits(self, qv: QuantizedVector) -> int:
        """(b-1)·d bits (MSE) + d bits (QJL signs) + 16 bits (norm) + 16 bits (γ)."""
        return self._mse_bits * self.d + self.d + 32

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Vectorized batch Algorithm 2 quantization."""
        n = len(X)
        x_norms = np.linalg.norm(X, axis=1, keepdims=True)  # (n, 1)
        safe_norms = np.maximum(x_norms, 1e-30)

        # --- MSE stage (vectorized) ---
        if self._mse_bits > 0:
            X_hat = X / safe_norms                           # (n, d)
            Y = X_hat @ self._rotation.T                     # (n, d)
            all_indices = np.searchsorted(self._boundaries, Y).astype(np.int8)
            Y_hat = self._centroids[all_indices]              # (n, d)
            if self._norm_correction:
                y_norms = np.linalg.norm(Y_hat, axis=1, keepdims=True)
                y_norms = np.maximum(y_norms, 1e-30)
                Y_hat = Y_hat / y_norms
            X_mse = x_norms * (Y_hat @ self._rotation)       # (n, d)
        else:
            all_indices = np.zeros((n, self.d), dtype=np.int8)
            X_mse = np.zeros_like(X)

        # --- QJL stage on residuals (vectorized) ---
        residuals = X - X_mse                                # (n, d)
        gammas = np.linalg.norm(residuals, axis=1)           # (n,)
        projections = residuals @ self._S.T                  # (n, d)
        all_signs = np.sign(projections).astype(np.int8)
        all_signs[all_signs == 0] = 1

        results = []
        for i in range(n):
            results.append(QuantizedVector(
                indices=all_indices[i],
                norms=np.array([x_norms[i, 0], gammas[i]], dtype=np.float32),
                signs=all_signs[i],
            ))
        return results

    def dequantize_batch(self, qvs: list[QuantizedVector]) -> np.ndarray:
        """Vectorized batch Algorithm 2 dequantization."""
        n = len(qvs)
        x_norms = np.array([float(qv.norms[0]) for qv in qvs])
        gammas = np.array([float(qv.norms[1]) for qv in qvs])

        # MSE reconstruction
        if self._mse_bits > 0:
            all_indices = np.array([qv.indices for qv in qvs])
            Y_hat = self._centroids[all_indices]              # (n, d)
            if self._norm_correction:
                y_norms = np.linalg.norm(Y_hat, axis=1, keepdims=True)
                y_norms = np.maximum(y_norms, 1e-30)
                Y_hat = Y_hat / y_norms
            X_mse = x_norms[:, np.newaxis] * (Y_hat @ self._rotation)
        else:
            X_mse = np.zeros((n, self.d))

        # QJL residual
        all_signs = np.array([qv.signs for qv in qvs], dtype=np.float64)
        R_hat = (QJL_CONST / self.d) * gammas[:, np.newaxis] * (all_signs @ self._S)

        return X_mse + R_hat
