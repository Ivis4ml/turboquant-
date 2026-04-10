"""
BlockTurboQuantMSE — block-wise TurboQuant with per-block fp16 scales.

Motivation
----------
Scalar TurboQuantMSE uses a single L2 norm per head-dim vector. On real
transformer K tensors this is catastrophically sensitive to outlier channels:
a handful of large-magnitude coordinates dominate the global norm, compress
the rest of the dynamic range into a tiny fraction of the codebook, and the
remaining coordinates end up poorly quantized.

llama.cpp's production `turbo*` path (and most int4/int8 LLM quantization)
fixes this by splitting each vector into blocks of 32 or 64 consecutive
coordinates and giving each block its own fp16 scale. Outlier coordinates
in one block no longer affect the centroids of another block.

This is the single most impactful structural change to close the gap between
the TurboQuant paper's Algorithm 1 (which this package implements faithfully
in `turboquant/mse.py`) and production-quality KV-cache compression.

Algorithm
---------
Given an input vector x ∈ R^d and block_size B with d = n_blocks * B:

Quantize:
  1. y = Π · x                          # Haar rotation (shared with other methods)
  2. y_blocks = y.reshape(n_blocks, B)
  3. for each block i:
       s_i = ||y_blocks[i]||            # fp16 scale
       y_blocks[i] /= s_i                # unit-norm block
  4. idx[i, j] = nearest_centroid(y_blocks[i, j], centroids_B)

Dequantize:
  1. y_blocks[i, j] = centroids_B[idx[i, j]]
  2. (optional norm correction: renormalize each block to unit norm)
  3. y_blocks[i] *= s_i
  4. y = y_blocks.flatten()
  5. x_hat = Π^T · y

Codebook
--------
The codebook is Lloyd-Max on N(0, 1/B) — NOT N(0, 1/d) — because after the
per-block unit-norm rescaling, each coordinate within a block is the j-th
coordinate of a B-dimensional unit vector, whose marginal distribution is
approximately N(0, 1/B) for B large enough. This is the same argument as
the scalar case but at the block level.

Storage
-------
Per vector: n_blocks * (B * num_bits + 16) bits
           = d * num_bits + 16 * n_blocks bits
           = (num_bits + 16/B) effective bits per dimension

For d=128, num_bits=4, block_size=32:
  576 bits/vector = 4.5 bits/dim = 3.56× vs fp16
  (vs scalar: 528 bits = 4.12 bits/dim = 3.88× vs fp16)

The block variant costs ~0.4 extra effective bits/dim in exchange for
(hypothetically) much better quality on real activations. This matches
turboquant_plus's `turbo4 = 4.25 bits/val` accounting.

Invariants
----------
- block_size must divide d evenly (we raise ValueError otherwise)
- block_size = d reduces exactly to scalar TurboQuantMSE (one block covers
  the whole vector). This is exercised by `test_block_quant.py`.
"""

from __future__ import annotations

import numpy as np

from vqbench.core.base import QuantizedVector, VectorQuantizer
from vqbench.core.rotation import haar_rotation
from vqbench.methods.turboquant.codebook import lloyd_max_codebook


class BlockTurboQuantMSE(VectorQuantizer):
    """
    Block-wise TurboQuantMSE with per-block fp16 scales.

    Args:
        d: Vector dimension (must be divisible by block_size).
        num_bits: Bits per coordinate for the block-local codebook.
        block_size: Size of each scalar quantization block (default 32).
                    Use block_size = d to recover scalar TurboQuantMSE.
        seed: Seed for the Haar rotation (shared with other methods for
              fairness — same (d, seed) gives the same rotation as
              `TurboQuantMSE`).
        norm_correction: Re-normalize each dequantized block to unit norm
                         before rescaling. Matches turboquant_plus production.
    """

    def __init__(
        self,
        d: int,
        num_bits: int,
        block_size: int = 32,
        seed: int = 42,
        norm_correction: bool = True,
    ) -> None:
        super().__init__(d=d, num_bits=num_bits, seed=seed)
        if d % block_size != 0:
            raise ValueError(
                f"d={d} must be divisible by block_size={block_size}"
            )
        self.block_size = block_size
        self.num_blocks = d // block_size
        self._norm_correction = norm_correction

        # Shared Haar rotation — same (d, seed) as scalar TurboQuantMSE
        self._rotation = haar_rotation(d, seed)

        # Block-local codebook: Lloyd-Max on N(0, 1/block_size)
        # NOTE: codebook dimension argument is block_size, not d.
        self._centroids, self._boundaries = lloyd_max_codebook(
            num_bits, block_size,
        )

    @property
    def name(self) -> str:
        return f"BlockTurboQuantMSE(B={self.block_size})"

    # ---------------------------------------------------------------------
    # Single-vector path
    # ---------------------------------------------------------------------

    def quantize(self, x: np.ndarray) -> QuantizedVector:
        # 1. Rotate (same as scalar TQ-MSE)
        y = self._rotation @ x.astype(np.float64)
        # 2. Split into blocks
        blocks = y.reshape(self.num_blocks, self.block_size)
        # 3. Per-block norm extraction
        scales = np.linalg.norm(blocks, axis=1)          # (num_blocks,)
        safe = np.maximum(scales, 1e-30)
        y_normed = blocks / safe[:, None]                # each block unit-norm
        # 4. Scalar quantize each block using the block-local codebook
        indices = np.searchsorted(self._boundaries, y_normed).astype(np.int8)

        return QuantizedVector(
            indices=indices.flatten(),
            # Store per-block scales as fp16 (2 bytes each)
            norms=scales.astype(np.float16),
            metadata={"block_size": self.block_size},
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        indices = qv.indices.reshape(self.num_blocks, self.block_size)
        scales = qv.norms.astype(np.float64)             # (num_blocks,)
        # 1. Lookup centroids block-by-block
        y_blocks = self._centroids[indices]              # (num_blocks, B)
        # 2. Optional per-block norm correction (turboquant_plus parity)
        if self._norm_correction:
            block_norms = np.linalg.norm(y_blocks, axis=1, keepdims=True)
            block_norms = np.maximum(block_norms, 1e-30)
            y_blocks = y_blocks / block_norms
        # 3. Rescale by per-block stored scales
        y_blocks = y_blocks * scales[:, None]
        # 4. Flatten and inverse rotate
        y = y_blocks.reshape(-1)
        return self._rotation.T @ y

    def storage_bits(self, qv: QuantizedVector) -> int:
        # Per vector: n_blocks * (block_size * num_bits) data bits
        #           + n_blocks * 16 fp16 scale bits
        data_bits = self.num_blocks * self.block_size * self.num_bits
        scale_bits = self.num_blocks * 16
        return data_bits + scale_bits

    # ---------------------------------------------------------------------
    # Batch paths — vectorized for speed
    # ---------------------------------------------------------------------

    def quantize_batch(self, X: np.ndarray) -> list[QuantizedVector]:
        """Batch quantization via (n, d) matmul + block reshape."""
        n = X.shape[0]
        # Batch rotate: (n, d)
        Y = X.astype(np.float64) @ self._rotation.T
        # (n, num_blocks, block_size)
        Y_blocks = Y.reshape(n, self.num_blocks, self.block_size)
        # Per-block scales per vector: (n, num_blocks)
        scales = np.linalg.norm(Y_blocks, axis=2)
        safe = np.maximum(scales, 1e-30)
        Y_normed = Y_blocks / safe[:, :, None]
        # Batch searchsorted: (n, num_blocks, block_size)
        all_indices = np.searchsorted(self._boundaries, Y_normed).astype(np.int8)

        results = []
        for i in range(n):
            results.append(QuantizedVector(
                indices=all_indices[i].flatten(),
                norms=scales[i].astype(np.float16),
                metadata={"block_size": self.block_size},
            ))
        return results

    def dequantize_batch(self, qvs: list[QuantizedVector]) -> np.ndarray:
        """Batch dequantization."""
        n = len(qvs)
        # (n, num_blocks, block_size)
        all_indices = np.stack([
            qv.indices.reshape(self.num_blocks, self.block_size) for qv in qvs
        ])
        all_scales = np.stack([
            qv.norms.astype(np.float64) for qv in qvs
        ])  # (n, num_blocks)
        Y_blocks = self._centroids[all_indices]          # (n, num_blocks, B)
        if self._norm_correction:
            block_norms = np.linalg.norm(Y_blocks, axis=2, keepdims=True)
            block_norms = np.maximum(block_norms, 1e-30)
            Y_blocks = Y_blocks / block_norms
        Y_blocks = Y_blocks * all_scales[:, :, None]
        Y = Y_blocks.reshape(n, self.d)
        return Y @ self._rotation                        # Π^T for row vectors
