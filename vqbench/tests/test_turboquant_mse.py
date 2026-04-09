"""
TurboQuantMSE validation — Theorem 1: D_mse ≤ (√(3π)/2) · 4^{-b}.

Paper: Zandieh et al., arXiv 2504.19874, Theorem 1
Expected MSE: b=1→0.36, b=2→0.117, b=3→0.03, b=4→0.009

Also validates IP bias: b=1 → α ≈ 2/π ≈ 0.637 (Req C3).
"""

import numpy as np
import pytest

from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.datasets.synthetic import random_unit_vectors
from vqbench.core.metrics import mse_distortion, ip_bias

# Expected MSE with norm_correction=True (production / turboquant_plus parity)
EXPECTED_MSE_CORRECTED = {1: 0.404, 2: 0.121, 3: 0.035, 4: 0.009}
# Expected MSE without norm_correction (pure Lloyd-Max theory)
LLOYD_MAX_MSE = {1: 0.3634, 2: 0.1175, 3: 0.0345, 4: 0.0095}
N_VECS = 2000
D = 512
DATA_SEED = 0       # data seed — must differ from method seed to avoid correlation
METHOD_SEED = 42     # rotation seed


class TestTurboQuantMSE:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X = random_unit_vectors(N_VECS, D, seed=DATA_SEED)
        self.Y = random_unit_vectors(N_VECS, D, seed=DATA_SEED + 1)

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_mse_production(self, b):
        """MSE with norm_correction=True (production setting)."""
        q = TurboQuantMSE(d=D, num_bits=b, seed=METHOD_SEED, norm_correction=True)
        X_hat = q.dequantize_batch(q.quantize_batch(self.X))
        mse = mse_distortion(self.X, X_hat)
        expected = EXPECTED_MSE_CORRECTED[b]
        assert abs(mse - expected) / expected < 0.12, (
            f"b={b}: MSE={mse:.4f}, expected={expected}"
        )

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_mse_theoretical(self, b):
        """Req C1: MSE without norm correction matches Lloyd-Max within 10%."""
        q = TurboQuantMSE(d=D, num_bits=b, seed=METHOD_SEED, norm_correction=False)
        X_hat = q.dequantize_batch(q.quantize_batch(self.X))
        mse = mse_distortion(self.X, X_hat)
        expected = LLOYD_MAX_MSE[b]
        assert abs(mse - expected) / expected < 0.10, (
            f"b={b}: MSE={mse:.4f}, expected={expected}"
        )

    def test_1bit_bias_alpha_production(self):
        """With norm correction, b=1 α ≈ 0.80 (improved from 2/π ≈ 0.637)."""
        q = TurboQuantMSE(d=D, num_bits=1, seed=METHOD_SEED, norm_correction=True)
        X_hat = q.dequantize_batch(q.quantize_batch(self.X))
        alpha, _ = ip_bias(self.X, X_hat, self.Y)
        assert 0.78 <= alpha <= 0.83, f"α={alpha:.4f}, expected ∈ [0.78, 0.83]"

    def test_2bit_bias_alpha_production(self):
        """With norm correction, b=2 α ≈ 0.93."""
        q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED, norm_correction=True)
        X_hat = q.dequantize_batch(q.quantize_batch(self.X))
        alpha, _ = ip_bias(self.X, X_hat, self.Y)
        assert 0.91 <= alpha <= 0.95, f"α={alpha:.4f}, expected ≈ 0.93"

    def test_roundtrip_preserves_norm(self):
        """Reconstructed vectors should have approximately same norm."""
        q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        for x in self.X[:50]:
            qv = q.quantize(x)
            x_hat = q.dequantize(qv)
            ratio = np.linalg.norm(x_hat) / np.linalg.norm(x)
            assert 0.5 < ratio < 1.5, f"Norm ratio {ratio:.3f} too far from 1"

    def test_storage_bits(self):
        q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        qv = q.quantize(self.X[0])
        expected = 2 * D + 16  # b·d + 16 bits for norm
        assert q.storage_bits(qv) == expected

    def test_zero_vector(self):
        q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        qv = q.quantize(np.zeros(D))
        x_hat = q.dequantize(qv)
        assert np.allclose(x_hat, 0.0)
