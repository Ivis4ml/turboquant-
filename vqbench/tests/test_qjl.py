"""
QJL unbiasedness and variance tests.

Paper: Zandieh et al., arXiv 2406.03482
  - Unbiased: E[⟨y, r̃⟩] = ⟨y, r⟩  (Lemma 4 of arXiv 2504.19874)
  - Variance: Var[⟨y, r̃⟩] ≤ (π/2d) · ‖y‖² · ‖r‖²
"""

import numpy as np
import pytest

from vqbench.methods.turboquant.qjl import QJLQuantizer


D = 256
N_TRIALS = 500


class TestQJL:
    def test_unbiased_ip(self):
        """E[⟨y, dequant(quant(x))⟩] ≈ ⟨y, x⟩ over many random S matrices."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal(D)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(D)
        y = y / np.linalg.norm(y)

        true_ip = np.dot(y, x)
        approx_ips = []
        for trial in range(N_TRIALS):
            q = QJLQuantizer(d=D, seed=trial)
            qv = q.quantize(x)
            x_hat = q.dequantize(qv)
            approx_ips.append(np.dot(y, x_hat))

        mean_ip = np.mean(approx_ips)
        bias = abs(mean_ip - true_ip)
        assert bias < 0.05, f"|bias| = {bias:.4f}, expected < 0.05"

    def test_variance_bound(self):
        """Var[⟨y, r̃⟩] ≤ (π/2d) · ‖y‖² · ‖r‖² (with margin)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(D) * 2.0  # non-unit
        y = rng.standard_normal(D) * 1.5

        approx_ips = []
        for trial in range(N_TRIALS):
            q = QJLQuantizer(d=D, seed=trial)
            qv = q.quantize(x)
            x_hat = q.dequantize(qv)
            approx_ips.append(np.dot(y, x_hat))

        var = np.var(approx_ips)
        bound = (np.pi / (2 * D)) * np.linalg.norm(y)**2 * np.linalg.norm(x)**2
        # Allow 50% margin for finite-sample variance
        assert var < bound * 1.5, f"Var={var:.6f}, bound={bound:.6f}"

    def test_storage_bits(self):
        q = QJLQuantizer(d=D, seed=42)
        qv = q.quantize(np.random.randn(D))
        assert q.storage_bits(qv) == D + 16

    def test_zero_vector(self):
        q = QJLQuantizer(d=D, seed=42)
        qv = q.quantize(np.zeros(D))
        x_hat = q.dequantize(qv)
        assert np.allclose(x_hat, 0.0)
