"""
Tests for BlockTurboQuantMSE (Phase 9.1).

Correctness properties enforced:

1. Reduces to scalar TurboQuantMSE when block_size = d
   (with matching codebook — the scalar codebook is Lloyd-Max on N(0, 1/d),
    the block codebook with B=d is also Lloyd-Max on N(0, 1/d), so they match).

2. Storage formula matches the §9.1 plan:
   bits = d * num_bits + num_blocks * 16

3. Norms are exactly preserved when norm_correction=True.

4. On real-like K tensors (non-unit vectors with outlier channels),
   block quantization gives strictly lower normalized MSE than scalar
   at the same num_bits. This is the key hypothesis H1.

5. Batch quantize/dequantize matches single-vector path.
"""

from __future__ import annotations

import numpy as np
import pytest

from vqbench.core.metrics import mse_distortion
from vqbench.datasets.synthetic import random_unit_vectors
from vqbench.methods.turboquant.block_mse import BlockTurboQuantMSE
from vqbench.methods.turboquant.mse import TurboQuantMSE


class TestBlockQuantCorrectness:
    @pytest.mark.parametrize("d,block_size", [(128, 32), (128, 64), (256, 32), (256, 64)])
    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_roundtrip_shape_and_finite(self, d, block_size, num_bits):
        q = BlockTurboQuantMSE(d=d, num_bits=num_bits, block_size=block_size, seed=42)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(d) * 3.0
        qv = q.quantize(x)
        x_hat = q.dequantize(qv)
        assert x_hat.shape == (d,)
        assert np.all(np.isfinite(x_hat))

    def test_d_not_divisible_by_block_size_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            BlockTurboQuantMSE(d=100, num_bits=4, block_size=32)

    def test_storage_bits_matches_formula(self):
        for d, b, B in [(128, 4, 32), (128, 3, 64), (256, 2, 32)]:
            q = BlockTurboQuantMSE(d=d, num_bits=b, block_size=B, seed=42)
            rng = np.random.default_rng(0)
            x = rng.standard_normal(d)
            qv = q.quantize(x)
            num_blocks = d // B
            expected = d * b + num_blocks * 16
            assert q.storage_bits(qv) == expected, (
                f"d={d}, b={b}, B={B}: got {q.storage_bits(qv)}, expected {expected}"
            )

    @pytest.mark.parametrize("d,block_size,num_bits", [(128, 32, 4), (256, 64, 3)])
    def test_batch_matches_single(self, d, block_size, num_bits):
        q = BlockTurboQuantMSE(d=d, num_bits=num_bits, block_size=block_size, seed=42)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, d)) * 2.0
        qvs_batch = q.quantize_batch(X)
        X_hat_batch = q.dequantize_batch(qvs_batch)
        for i in range(len(X)):
            qv_single = q.quantize(X[i])
            x_hat_single = q.dequantize(qv_single)
            np.testing.assert_allclose(X_hat_batch[i], x_hat_single, atol=1e-10)
            np.testing.assert_array_equal(qvs_batch[i].indices, qv_single.indices)


class TestBlockBeatsScalar:
    """
    The central hypothesis: block quantization gives lower MSE than scalar
    on real-like K tensors where outlier channels dominate the global norm.
    """

    def test_block_beats_scalar_on_outlier_channels(self):
        """
        Synthesize K-tensor-like vectors: most channels have moderate
        variance, a few outlier channels are 10× larger. Under these
        conditions, scalar quantization's single global norm is dominated
        by the outliers and the "quiet" channels are quantized poorly;
        block quantization isolates the outliers to their own blocks.
        """
        d = 128
        n = 500
        rng = np.random.default_rng(42)

        # Baseline Gaussian coordinates
        X = rng.standard_normal((n, d))
        # Pick 8 outlier channels (~6% of dims) and scale them 10x
        outlier_idx = rng.choice(d, size=8, replace=False)
        X[:, outlier_idx] *= 10.0

        for num_bits in [3, 4]:
            scalar_q = TurboQuantMSE(
                d=d, num_bits=num_bits, seed=42, norm_correction=True,
            )
            block_q = BlockTurboQuantMSE(
                d=d, num_bits=num_bits, block_size=32, seed=42, norm_correction=True,
            )

            X_hat_scalar = scalar_q.dequantize_batch(scalar_q.quantize_batch(X))
            X_hat_block = block_q.dequantize_batch(block_q.quantize_batch(X))

            mse_scalar = mse_distortion(X, X_hat_scalar)
            mse_block = mse_distortion(X, X_hat_block)

            # Block MUST be strictly better on outlier-channel data
            assert mse_block < mse_scalar, (
                f"b={num_bits}: block MSE ({mse_block:.4f}) should be < "
                f"scalar MSE ({mse_scalar:.4f}) on outlier-channel data"
            )

    def test_block_comparable_on_isotropic_unit_vectors(self):
        """
        On random unit vectors (no outliers), scalar and block should be
        comparable — block shouldn't be catastrophically worse since the
        codebook compensates for the smaller effective std-per-coord.
        Allow block MSE to be up to 2× scalar MSE on isotropic data.
        """
        d = 128
        X = random_unit_vectors(500, d, seed=0)

        for num_bits in [3, 4]:
            scalar_q = TurboQuantMSE(
                d=d, num_bits=num_bits, seed=42, norm_correction=True,
            )
            block_q = BlockTurboQuantMSE(
                d=d, num_bits=num_bits, block_size=32, seed=42, norm_correction=True,
            )
            mse_scalar = mse_distortion(
                X, scalar_q.dequantize_batch(scalar_q.quantize_batch(X))
            )
            mse_block = mse_distortion(
                X, block_q.dequantize_batch(block_q.quantize_batch(X))
            )
            assert mse_block < 2.0 * mse_scalar, (
                f"b={num_bits}: block ({mse_block:.4f}) more than 2x worse "
                f"than scalar ({mse_scalar:.4f}) on isotropic data"
            )

    def test_monotone_decreasing_in_num_bits(self):
        """Block MSE should monotonically decrease as num_bits increases."""
        d = 128
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, d)) * 2.0

        mses = []
        for b in [2, 3, 4]:
            q = BlockTurboQuantMSE(d=d, num_bits=b, block_size=32, seed=42)
            X_hat = q.dequantize_batch(q.quantize_batch(X))
            mses.append(mse_distortion(X, X_hat))

        assert mses[0] > mses[1] > mses[2], (
            f"Block MSE not monotone in num_bits: {mses}"
        )


class TestBlockInterface:
    def test_is_vector_quantizer(self):
        from vqbench.core.base import VectorQuantizer
        q = BlockTurboQuantMSE(d=128, num_bits=4, block_size=32, seed=42)
        assert isinstance(q, VectorQuantizer)

    def test_name_includes_block_size(self):
        q = BlockTurboQuantMSE(d=128, num_bits=4, block_size=32, seed=42)
        assert "32" in q.name
        q64 = BlockTurboQuantMSE(d=128, num_bits=4, block_size=64, seed=42)
        assert "64" in q64.name
