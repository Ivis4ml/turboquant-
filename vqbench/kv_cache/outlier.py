"""
Outlier channel strategy: mixed-precision quantization.

Req F7: 2.5-bit / 3.5-bit mixed precision, shared across methods.

Key idea: a small fraction of channels (dimensions) have much larger magnitude
than others. These "outlier" channels contribute disproportionately to MSE.
Strategy: use full precision for outlier channels, quantize the rest.
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import VectorQuantizer, QuantizedVector


class OutlierAwareQuantizer(VectorQuantizer):
    """
    Wrapper that keeps outlier channels in full precision.

    Mixed precision: outlier channels stored in fp16 (16 bits each),
    remaining channels quantized at `num_bits` bits per dim.

    Effective bits per dim: num_bits + (16 - num_bits) * outlier_fraction
    e.g., 2-bit + 5% fp16 outliers ≈ 2.7 bits/dim ("2.5-bit")
         3-bit + 5% fp16 outliers ≈ 3.65 bits/dim ("3.5-bit")
    """

    def __init__(
        self,
        base_quantizer: VectorQuantizer,
        outlier_fraction: float = 0.05,
    ) -> None:
        super().__init__(
            d=base_quantizer.d,
            num_bits=base_quantizer.num_bits,
            seed=base_quantizer.seed,
        )
        self._base = base_quantizer
        self._outlier_fraction = outlier_fraction
        self._outlier_mask: np.ndarray | None = None  # bool array (d,)

    @property
    def name(self) -> str:
        return f"{self._base.name}+outlier"

    def detect_outliers(self, X: np.ndarray) -> None:
        """
        Detect outlier channels from training data.

        Strategy: channels with highest variance across the dataset are outliers.
        """
        channel_var = np.var(X, axis=0)  # (d,)
        n_outlier = max(1, int(self.d * self._outlier_fraction))
        threshold_idx = np.argsort(channel_var)[-n_outlier:]
        self._outlier_mask = np.zeros(self.d, dtype=bool)
        self._outlier_mask[threshold_idx] = True

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        if self._outlier_mask is None:
            return self._base.quantize(x)

        # Zero out outlier channels for base quantizer
        x_masked = x.copy()
        outlier_values = x[self._outlier_mask].copy()
        x_masked[self._outlier_mask] = 0.0

        qv = self._base.quantize(x_masked)
        qv.metadata["outlier_values"] = outlier_values.astype(np.float16)
        qv.metadata["outlier_mask"] = self._outlier_mask
        return qv

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        x_hat = self._base.dequantize(qv)
        if "outlier_values" in qv.metadata:
            mask = qv.metadata["outlier_mask"]
            x_hat[mask] = qv.metadata["outlier_values"].astype(np.float64)
        return x_hat

    def storage_bits(self, qv: QuantizedVector) -> int:
        base_bits = self._base.storage_bits(qv)
        if self._outlier_mask is not None:
            n_outlier = int(self._outlier_mask.sum())
            # Outlier channels: 16 bits each (fp16) instead of num_bits
            outlier_extra = n_outlier * (16 - self.num_bits)
            return base_bits + outlier_extra
        return base_bits
