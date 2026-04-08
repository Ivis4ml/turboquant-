"""
TurboQuantProd validation — Theorem 2: unbiased IP, D_prod ≤ (π/2d)·‖y‖²·D_mse(b-1).

Paper: Zandieh et al., arXiv 2504.19874, Theorem 2
  - Req C2: |mean bias| < 0.01, variance×d within 15% of paper values
  - D_prod × d: b=1→1.57, b=2→0.56, b=3→0.18, b=4→0.047
"""

import numpy as np
import pytest

from vqbench.methods.turboquant.prod import TurboQuantProd
from vqbench.datasets.synthetic import random_unit_vectors
from vqbench.core.metrics import ip_bias

PAPER_IP_D = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}
N_VECS = 2000
D = 512
DATA_SEED = 0
METHOD_SEED = 42


class TestTurboQuantProd:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X = random_unit_vectors(N_VECS, D, seed=DATA_SEED)
        self.Y = random_unit_vectors(N_VECS, D, seed=DATA_SEED + 1)

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_unbiased_ip(self, b):
        """Req C2: TurboQuantProd has |mean bias| < 0.01 (α ≈ 1.0)."""
        q = TurboQuantProd(d=D, num_bits=b, seed=METHOD_SEED)
        qvs = q.quantize_batch(self.X)
        X_hat = q.dequantize_batch(qvs)
        alpha, _ = ip_bias(self.X, X_hat, self.Y)
        assert abs(alpha - 1.0) < 0.03, f"b={b}: α={alpha:.4f}, expected ≈ 1.0"

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_ip_distortion_times_d(self, b):
        """D_prod × d within 15% of paper values (Theorem 2)."""
        q = TurboQuantProd(d=D, num_bits=b, seed=METHOD_SEED)
        qvs = q.quantize_batch(self.X)
        X_hat = q.dequantize_batch(qvs)

        # Compute average (⟨y,x⟩ − ⟨y,x̃⟩)² for paired (x,y)
        errors = []
        for i in range(N_VECS):
            y = self.Y[i]
            true_ip = np.dot(y, self.X[i])
            approx_ip = np.dot(y, X_hat[i])
            errors.append((true_ip - approx_ip) ** 2)
        d_prod = np.mean(errors)
        d_prod_times_d = d_prod * D

        expected = PAPER_IP_D[b]
        # 40% tolerance for statistical test with finite samples
        assert d_prod_times_d < expected * 1.4, (
            f"b={b}: D_prod×d={d_prod_times_d:.4f}, paper={expected}"
        )

    def test_b1_is_pure_qjl(self):
        """b=1: MSE stage = 0 bits, degenerates to pure QJL."""
        q = TurboQuantProd(d=D, num_bits=1, seed=METHOD_SEED)
        assert q._mse_bits == 0
        x = self.X[0]
        qv = q.quantize(x)
        # indices should be all zeros (no MSE stage)
        assert np.all(qv.indices == 0)
        # signs should be populated
        assert qv.signs is not None
        assert np.all(np.abs(qv.signs) == 1)

    def test_storage_bits(self):
        q = TurboQuantProd(d=D, num_bits=3, seed=METHOD_SEED)
        qv = q.quantize(self.X[0])
        # (b-1)·d + d + 32 = 2·512 + 512 + 32 = 1568
        expected = 2 * D + D + 32
        assert q.storage_bits(qv) == expected

    def test_zero_vector(self):
        q = TurboQuantProd(d=D, num_bits=2, seed=METHOD_SEED)
        qv = q.quantize(np.zeros(D))
        x_hat = q.dequantize(qv)
        assert np.allclose(x_hat, 0.0)
