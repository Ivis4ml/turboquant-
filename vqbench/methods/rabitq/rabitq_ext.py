"""
Extended RaBitQ: B-bit quantization via bit-plane decomposition.

Paper: Gao et al., arXiv 2409.09913, §3
Pipeline:
    x → centroid subtraction → normalize → rotate (same as RaBitQ 1-bit)
    → B-bit integer quantize: idx[j] ∈ {0, ..., 2^B − 1}
    → Integer codebook: c_k = (2k − 2^B + 1)  for k = 0..2^B−1
    → Store: B·d bits, ‖o‖, ip_coeff, scale, offset

Invariant: B=1 must produce identical results to RaBitQ1Bit.

Why NOT Lloyd-Max: RaBitQ's unbiased estimator requires linear algebraic structure.
The B-bit code decomposes into B binary planes, each computable via popcount.
Lloyd-Max centroids don't decompose this way.
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.core.rotation import haar_rotation

RABITQ_EXPECTED_IP_COEFF = np.sqrt(2 / np.pi)


def ext_rabitq_codebook(num_bits: int) -> np.ndarray:
    """
    Integer grid codebook for Extended RaBitQ.

    Paper: arXiv 2409.09913, §3.1
    B=1: [-1, 1], B=2: [-3, -1, 1, 3], B=3: [-7,-5,-3,-1,1,3,5,7]

    Returns:
        Centered integer levels of shape (2^B,).
    """
    levels = np.arange(2 ** num_bits)
    return 2 * levels - (2 ** num_bits - 1)          # ← Eq. (3)


class ExtRaBitQ(VectorQuantizer):
    """
    Extended RaBitQ: B-bit with bit-plane decomposition.

    Paper: Gao et al., arXiv 2409.09913
    """

    def __init__(self, d: int, num_bits: int, seed: int = 42) -> None:
        super().__init__(d, num_bits, seed)
        self._rotation = haar_rotation(d, seed)
        self._codebook = ext_rabitq_codebook(num_bits).astype(np.float64)
        self._centroid: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "ExtRaBitQ"

    def fit(self, X: np.ndarray) -> None:
        """Compute dataset centroid."""
        self._centroid = np.mean(X, axis=0)

    def _get_centroid(self, d: int) -> np.ndarray:
        if self._centroid is not None:
            return self._centroid
        return np.zeros(d)

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """
        ExtRaBitQ quantization.

        Paper: arXiv 2409.09913, §3.1
        Same preprocessing as RaBitQ 1-bit, then B-bit integer quantization.
        """
        centroid = self._get_centroid(self.d)
        o = x - centroid
        norm_o = np.linalg.norm(o)

        if norm_o < 1e-30:
            return QuantizedVector(
                indices=np.zeros(self.d, dtype=np.int8),
                norms=np.array([0.0], dtype=np.float32),
                metadata={
                    "ip_coeff": RABITQ_EXPECTED_IP_COEFF,
                    "centroid": centroid,
                    "scale": 1.0,
                    "offset": 0.0,
                },
            )

        o_bar = o / norm_o
        y = self._rotation @ o_bar                   # ← rotate

        # Scale y to the codebook range for optimal quantization
        # Codebook: [-2^B+1, ..., 2^B-1] with step 2
        # Map y ∈ approx [-3σ, 3σ] to codebook range
        sigma = 1.0 / np.sqrt(self.d)
        k = 2 ** self.num_bits
        half_range = k - 1  # max codebook value

        # Scale factor: map ±3σ to ±half_range
        scale = half_range / (3.0 * sigma)
        scaled = y * scale                           # ← scale to codebook range

        # Quantize to nearest codebook entry
        # Codebook entries are odd integers: -half_range, ..., -1, 1, ..., half_range
        # Nearest odd integer: round to nearest odd
        rounded = np.round((scaled - 1) / 2) * 2 + 1
        rounded = np.clip(rounded, -half_range, half_range)
        indices = ((rounded + half_range) / 2).astype(np.int8)

        # Compute ip_coeff for unbiased estimation
        # x̄ = codebook[indices] (normalized)
        x_bar_raw = self._codebook[indices]
        x_bar_norm = np.linalg.norm(x_bar_raw)
        if x_bar_norm < 1e-30:
            ip_coeff = RABITQ_EXPECTED_IP_COEFF
        else:
            x_bar = x_bar_raw / x_bar_norm
            ip_coeff = float(np.dot(x_bar, y))

        return QuantizedVector(
            indices=indices,
            norms=np.array([norm_o], dtype=np.float32),
            metadata={
                "ip_coeff": ip_coeff,
                "centroid": centroid,
                "scale": 1.0 / scale,  # inverse for dequantization
                "offset": 0.0,
            },
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """
        ExtRaBitQ dequantization.

        Map indices back to codebook values, unscale, inverse rotate, add centroid.
        """
        norm_o = float(qv.norms[0])
        centroid = qv.metadata.get("centroid", np.zeros(self.d))

        if norm_o < 1e-30:
            return centroid.copy()

        # Reconstruct rotated coordinates
        y_raw = self._codebook[qv.indices]           # ← integer codebook values
        scale_inv = qv.metadata["scale"]             # ← stored inverse scale
        y_hat = y_raw * scale_inv                    # ← unscale to original range

        # Normalize to unit vector (since o_bar was unit)
        y_norm = np.linalg.norm(y_hat)
        if y_norm > 1e-30:
            y_hat = y_hat / y_norm

        x_hat = norm_o * (self._rotation.T @ y_hat)
        return x_hat + centroid

    def storage_bits(self, qv: QuantizedVector) -> int:
        """B·d bits + 32+32+32+32 bits (norm + ip_coeff + scale + offset)."""
        return self.num_bits * self.d + 128

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Vectorized batch quantization."""
        centroid = self._get_centroid(self.d)
        O = X - centroid[np.newaxis, :]
        norms = np.linalg.norm(O, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-30)
        O_bar = O / safe_norms
        Y = O_bar @ self._rotation.T

        sigma = 1.0 / np.sqrt(self.d)
        k = 2 ** self.num_bits
        half_range = k - 1
        scale = half_range / (3.0 * sigma)

        scaled = Y * scale
        rounded = np.round((scaled - 1) / 2) * 2 + 1
        rounded = np.clip(rounded, -half_range, half_range)
        all_indices = ((rounded + half_range) / 2).astype(np.int8)

        results = []
        for i in range(len(X)):
            indices = all_indices[i]
            x_bar_raw = self._codebook[indices]
            x_bar_norm = np.linalg.norm(x_bar_raw)
            if x_bar_norm < 1e-30:
                ip_coeff = RABITQ_EXPECTED_IP_COEFF
            else:
                x_bar = x_bar_raw / x_bar_norm
                ip_coeff = float(np.dot(x_bar, Y[i]))

            results.append(QuantizedVector(
                indices=indices,
                norms=np.array([norms[i, 0]], dtype=np.float32),
                metadata={
                    "ip_coeff": ip_coeff,
                    "centroid": centroid,
                    "scale": 1.0 / scale,
                    "offset": 0.0,
                },
            ))
        return results
