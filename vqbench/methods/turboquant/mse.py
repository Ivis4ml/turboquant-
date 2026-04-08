"""
TurboQuantMSE — Algorithm 1: MSE-optimal scalar quantization.

Paper: Zandieh et al., arXiv 2504.19874, Algorithm 1
Pipeline:
    x ∈ R^d → store ‖x‖ → x̂ = x/‖x‖ → y = Π·x̂
    → idx[j] = nearest centroid(y[j]) → store idx
    DeQuant: ỹ[j] = codebook[idx[j]] → x̃ = ‖x‖ · Πᵀ·ỹ

MSE guarantee (Theorem 1): D_mse ≤ (√(3π)/2) · 4^{-b}

IP bias: E[⟨y, x̃⟩] = α · ⟨y, x⟩ where α < 1
    b=1: α = 2/π ≈ 0.637
    b=2: α ≈ 0.883
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.core.rotation import haar_rotation
from vqbench.methods.turboquant.codebook import lloyd_max_codebook


class TurboQuantMSE(VectorQuantizer):
    """
    TurboQuant Algorithm 1: random rotation + Lloyd-Max scalar quantization.

    Paper: Zandieh et al., arXiv 2504.19874, Algorithm 1
    """

    def __init__(self, d: int, num_bits: int, seed: int = 42) -> None:
        super().__init__(d, num_bits, seed)
        self._rotation = haar_rotation(d, seed)
        self._centroids, self._boundaries = lloyd_max_codebook(num_bits, d)

    @property
    def name(self) -> str:
        return "TurboQuantMSE"

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """
        Algorithm 1 quantization.

        Paper: arXiv 2504.19874, Algorithm 1, lines 1-4
        x → ‖x‖, x̂ = x/‖x‖ → y = Π·x̂ → idx = nearest centroid per coord
        """
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-30:
            return QuantizedVector(
                indices=np.zeros(self.d, dtype=np.int8),
                norms=np.array([0.0], dtype=np.float32),
            )

        x_hat = x / x_norm                          # ← normalize
        y = self._rotation @ x_hat                   # ← rotate: y = Π·x̂
        # Quantize each coordinate to nearest centroid via searchsorted
        # boundaries are sorted, searchsorted gives the right bin
        indices = np.searchsorted(self._boundaries, y).astype(np.int8)
        return QuantizedVector(
            indices=indices,
            norms=np.array([x_norm], dtype=np.float32),
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """
        Algorithm 1 dequantization.

        Paper: arXiv 2504.19874, Algorithm 1, lines 5-6
        ỹ[j] = codebook[idx[j]] → x̃ = ‖x‖ · Πᵀ·ỹ
        """
        x_norm = float(qv.norms[0])
        if x_norm < 1e-30:
            return np.zeros(self.d)

        y_hat = self._centroids[qv.indices]          # ← lookup centroids
        x_hat = self._rotation.T @ y_hat             # ← inverse rotate: Πᵀ·ỹ
        return x_norm * x_hat                        # ← rescale

    def storage_bits(self, qv: QuantizedVector) -> int:
        """b·d bits (indices) + 16 bits (fp16 norm)."""
        return self.num_bits * self.d + 16

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Vectorized batch quantization."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-30)
        X_hat = X / safe_norms                       # (n, d)
        Y = X_hat @ self._rotation.T                 # (n, d) — Y = (Π·x̂ᵀ)ᵀ = X̂·Πᵀ
        # Batch searchsorted
        indices = np.searchsorted(self._boundaries, Y).astype(np.int8)
        results = []
        for i in range(len(X)):
            results.append(QuantizedVector(
                indices=indices[i],
                norms=np.array([norms[i, 0]], dtype=np.float32),
            ))
        return results
