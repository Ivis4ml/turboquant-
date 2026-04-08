"""
VectorQuantizer ABC and QuantizedVector dataclass.

Every quantization method in VQBench implements VectorQuantizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuantizedVector:
    """Container for a quantized vector's compressed representation.

    Attributes:
        indices: Quantization indices (int array). Interpretation is method-specific.
        norms: Stored scalars — norms, gamma, scale factors, etc.
        signs: QJL sign bits (TurboQuantProd only); None for other methods.
        metadata: Method-specific extras (ip_coeff, centroid, etc.).
    """

    indices: np.ndarray
    norms: np.ndarray
    signs: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class VectorQuantizer(ABC):
    """Abstract base class for all vector quantization methods.

    Args:
        d: Vector dimensionality.
        num_bits: Bits per dimension for quantization.
        seed: Random seed for reproducibility.
    """

    def __init__(self, d: int, num_bits: int, seed: int = 42) -> None:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")
        self.d = d
        self.num_bits = num_bits
        self.seed = seed

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name (e.g. 'TurboQuantMSE')."""

    @abstractmethod
    def quantize(self, x: np.ndarray) -> QuantizedVector:
        """Quantize a single vector x ∈ R^d."""

    @abstractmethod
    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """Reconstruct an approximate vector from its quantized form."""

    @abstractmethod
    def storage_bits(self, qv: QuantizedVector) -> int:
        """Total bits used to store this quantized vector (indices + metadata)."""

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Quantize a batch X ∈ R^{n×d}. Override for vectorized implementations."""
        return [self.quantize(x) for x in X]

    def dequantize_batch(self, qvs: list[QuantizedVector]) -> np.ndarray:
        """Dequantize a batch. Override for vectorized implementations."""
        return np.array([self.dequantize(qv) for qv in qvs])
