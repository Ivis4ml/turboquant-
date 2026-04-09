"""
QJL — Quantized Johnson-Lindenstrauss transform.

Paper: Zandieh et al., arXiv 2406.03482, Definition 1
Formula:
    Quantize: qjl = sign(S · r),  S ∈ R^{d×d}, S_{ij} ~ N(0,1)
    DeQuant:  r̃ = √(π/2)/d · γ · Sᵀ · qjl,  where γ = ‖r‖

Properties:
    - Unbiased: E[⟨y, r̃⟩] = ⟨y, r⟩  (Lemma 4 of arXiv 2504.19874)
    - Variance: Var[⟨y, r̃⟩] ≤ (π/2d) · ‖y‖² · ‖r‖²
    - Storage: d bits (signs) + 1 scalar (γ)
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector

# √(π/2) ≈ 1.2533                                  ← Definition 1
QJL_CONST = np.sqrt(np.pi / 2)


class QJLQuantizer(VectorQuantizer):
    """
    QJL 1-bit quantizer with unbiased inner product recovery.

    Paper: Zandieh et al., arXiv 2406.03482
    Used as a building block in TurboQuantProd (Algorithm 2).
    """

    def __init__(self, d: int, num_bits: int = 1, seed: int = 42) -> None:
        super().__init__(d, num_bits=1, seed=seed)
        # S matrix: S_{ij} ~ N(0,1), independent from rotation Π
        # Use a different seed stream to ensure independence
        rng = np.random.default_rng(seed + 2**31)
        self._S = rng.standard_normal((d, d))

    @property
    def name(self) -> str:
        return "QJL"

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """
        QJL quantization: sign(S · x).

        Paper: arXiv 2406.03482, Definition 1
        """
        gamma = np.linalg.norm(x)                   # ← γ = ‖x‖
        if gamma < 1e-30:
            return QuantizedVector(
                indices=np.zeros(self.d, dtype=np.int8),
                norms=np.array([0.0], dtype=np.float32),
                signs=np.ones(self.d, dtype=np.int8),
            )

        proj = self._S @ x                          # ← S · x
        signs = np.sign(proj).astype(np.int8)        # ← sign(S · x)
        signs[signs == 0] = 1                        # tie-break

        return QuantizedVector(
            indices=np.zeros(self.d, dtype=np.int8),  # unused for QJL
            norms=np.array([gamma], dtype=np.float32),
            signs=signs,
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """
        QJL dequantization.

        Paper: arXiv 2406.03482, Definition 1
        Formula: x̃ = √(π/2) / d · γ · Sᵀ · sign(S·x)
        """
        gamma = float(qv.norms[0])
        if gamma < 1e-30:
            return np.zeros(self.d)

        # √(π/2) / d · γ · Sᵀ · z                  ← Definition 1
        return QJL_CONST / self.d * gamma * (self._S.T @ qv.signs.astype(np.float64))

    def storage_bits(self, qv: QuantizedVector) -> int:
        """d bits (signs) + 16 bits (gamma fp16)."""
        return self.d + 16

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Vectorized batch QJL quantization."""
        gammas = np.linalg.norm(X, axis=1)           # (n,)
        projections = X @ self._S.T                   # (n, d) — batch S·x
        all_signs = np.sign(projections).astype(np.int8)
        all_signs[all_signs == 0] = 1
        results = []
        for i in range(len(X)):
            results.append(QuantizedVector(
                indices=np.zeros(self.d, dtype=np.int8),
                norms=np.array([gammas[i]], dtype=np.float32),
                signs=all_signs[i],
            ))
        return results

    def dequantize_batch(self, qvs: list[QuantizedVector]) -> np.ndarray:
        """Vectorized batch QJL dequantization."""
        n = len(qvs)
        gammas = np.array([float(qv.norms[0]) for qv in qvs])     # (n,)
        signs = np.array([qv.signs for qv in qvs], dtype=np.float64)  # (n, d)
        # x̃_i = √(π/2)/d · γ_i · Sᵀ · signs_i
        # batch: X̃ = (√(π/2)/d) · diag(γ) · (signs @ S)
        out = (QJL_CONST / self.d) * (signs @ self._S) * gammas[:, np.newaxis]
        return out
