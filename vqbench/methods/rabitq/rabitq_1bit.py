"""
RaBitQ 1-bit: hypercube quantization + unbiased estimator.

Paper: Gao & Long, arXiv 2405.12497, Algorithm 1
Pipeline:
    x → centroid subtraction: o = x − c
    → normalize: ō = o/‖o‖, store ‖o‖
    → rotate: y = P · ō
    → hypercube quantize: signs = sign(y) ∈ {−1,+1}^d
    → normalized: x̄ = signs/√d  (unit hypercube vertex)
    → store: ip_coeff = ⟨x̄, y⟩ = Σ(signs·y)/√d
    → store: sign bits (d bits), ‖o‖, ip_coeff

Key difference from TurboQuant at 1-bit:
    RaBitQ centroids: ±1/√d  (hypercube)
    TurboQuant centroids: ±0.7979/√d  (Lloyd-Max optimal)
    RaBitQ MSE is worse, but IP estimation is natively unbiased.
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.core.rotation import haar_rotation

# Expected value of ip_coeff for large d                 ← §3.2
RABITQ_EXPECTED_IP_COEFF = np.sqrt(2 / np.pi)  # ≈ 0.7979


class RaBitQ1Bit(VectorQuantizer):
    """
    RaBitQ 1-bit quantizer with unbiased distance estimation.

    Paper: Gao & Long, arXiv 2405.12497
    """

    def __init__(self, d: int, num_bits: int = 1, seed: int = 42) -> None:
        super().__init__(d, num_bits=1, seed=seed)
        self._rotation = haar_rotation(d, seed)
        self._centroid: np.ndarray | None = None  # set via fit()

    @property
    def name(self) -> str:
        return "RaBitQ1Bit"

    def fit(self, X: np.ndarray) -> None:
        """
        Compute dataset centroid for centroid subtraction.

        Paper: arXiv 2405.12497, §3.1
        For random unit vectors, c ≈ 0; for real data, significant.
        """
        self._centroid = np.mean(X, axis=0)

    def _get_centroid(self, d: int) -> np.ndarray:
        if self._centroid is not None:
            return self._centroid
        return np.zeros(d)

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """
        RaBitQ 1-bit quantization.

        Paper: arXiv 2405.12497, Algorithm 1
        o = x − c → ō = o/‖o‖ → y = P·ō → signs = sign(y)
        → ip_coeff = ⟨signs/√d, y⟩
        """
        centroid = self._get_centroid(self.d)
        o = x - centroid                             # ← centroid subtraction
        norm_o = np.linalg.norm(o)

        if norm_o < 1e-30:
            return QuantizedVector(
                indices=np.ones(self.d, dtype=np.int8),
                norms=np.array([0.0], dtype=np.float32),
                metadata={"ip_coeff": RABITQ_EXPECTED_IP_COEFF, "centroid": centroid},
            )

        o_bar = o / norm_o                           # ← normalize
        y = self._rotation @ o_bar                   # ← rotate: P·ō

        signs = np.sign(y).astype(np.int8)           # ← hypercube quantize
        signs[signs == 0] = 1                        # tie-break

        # ip_coeff = ⟨x̄, y⟩ where x̄ = signs/√d     ← §3.2
        x_bar = signs.astype(np.float64) / np.sqrt(self.d)
        ip_coeff = float(np.dot(x_bar, y))

        return QuantizedVector(
            indices=signs,  # ±1 sign bits
            norms=np.array([norm_o], dtype=np.float32),
            metadata={"ip_coeff": ip_coeff, "centroid": centroid},
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """
        RaBitQ dequantization (direct, not via estimator).

        ỹ = signs/√d → x̃ = ‖o‖ · Pᵀ · ỹ + c
        """
        norm_o = float(qv.norms[0])
        if norm_o < 1e-30:
            return qv.metadata.get("centroid", np.zeros(self.d)).copy()

        y_hat = qv.indices.astype(np.float64) / np.sqrt(self.d)
        x_hat = norm_o * (self._rotation.T @ y_hat)
        centroid = qv.metadata.get("centroid", np.zeros(self.d))
        return x_hat + centroid

    def storage_bits(self, qv: QuantizedVector) -> int:
        """d bits (signs) + 32 bits (fp32 ‖o‖) + 32 bits (fp32 ip_coeff)."""
        return self.d + 64

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Vectorized batch quantization."""
        centroid = self._get_centroid(self.d)
        O = X - centroid[np.newaxis, :]              # (n, d)
        norms = np.linalg.norm(O, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-30)
        O_bar = O / safe_norms                       # (n, d)
        Y = O_bar @ self._rotation.T                 # (n, d)

        signs = np.sign(Y).astype(np.int8)
        signs[signs == 0] = 1

        # ip_coeff per vector: ⟨signs_i/√d, Y_i⟩
        x_bar = signs.astype(np.float64) / np.sqrt(self.d)
        ip_coeffs = np.sum(x_bar * Y, axis=1)       # (n,)

        results = []
        for i in range(len(X)):
            results.append(QuantizedVector(
                indices=signs[i],
                norms=np.array([norms[i, 0]], dtype=np.float32),
                metadata={"ip_coeff": float(ip_coeffs[i]), "centroid": centroid},
            ))
        return results
