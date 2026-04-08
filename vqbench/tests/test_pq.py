"""
PQ and OPQ tests.

PQ is the only offline (data-dependent) method — requires training.
OPQ should achieve equal or lower MSE than PQ via learned rotation.
"""

import numpy as np
import pytest

from vqbench.methods.pq.product_quant import ProductQuantizer
from vqbench.methods.pq.opq import OptimizedPQ
from vqbench.datasets.synthetic import random_unit_vectors
from vqbench.core.metrics import mse_distortion

N_TRAIN = 1000
N_TEST = 500
D = 128  # smaller d for PQ speed
DATA_SEED = 0
METHOD_SEED = 42


class TestPQ:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X_train = random_unit_vectors(N_TRAIN, D, seed=DATA_SEED)
        self.X_test = random_unit_vectors(N_TEST, D, seed=DATA_SEED + 10)

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_mse_decreases_with_bits(self, b):
        """More bits → lower MSE."""
        if b == 2:
            pytest.skip("Need baseline for comparison")
        q_lo = ProductQuantizer(d=D, num_bits=b - 1, seed=METHOD_SEED)
        q_hi = ProductQuantizer(d=D, num_bits=b, seed=METHOD_SEED)
        q_lo.fit(self.X_train)
        q_hi.fit(self.X_train)

        X_hat_lo = q_lo.dequantize_batch(q_lo.quantize_batch(self.X_test))
        X_hat_hi = q_hi.dequantize_batch(q_hi.quantize_batch(self.X_test))

        mse_lo = mse_distortion(self.X_test, X_hat_lo)
        mse_hi = mse_distortion(self.X_test, X_hat_hi)
        assert mse_hi <= mse_lo + 0.01, f"b={b}: MSE={mse_hi:.4f} not better than b-1 MSE={mse_lo:.4f}"

    def test_untrained_raises(self):
        """Quantize before fit() should raise."""
        q = ProductQuantizer(d=D, num_bits=2, seed=METHOD_SEED)
        with pytest.raises(RuntimeError, match="trained"):
            q.quantize(self.X_test[0])

    def test_roundtrip(self):
        """Quantize → dequantize should produce reasonable reconstruction."""
        q = ProductQuantizer(d=D, num_bits=2, seed=METHOD_SEED)
        q.fit(self.X_train)
        x = self.X_test[0]
        qv = q.quantize(x)
        x_hat = q.dequantize(qv)
        mse = np.sum((x - x_hat) ** 2)
        assert mse < 1.0, f"Single-vector MSE={mse:.4f} too high"

    def test_adc(self):
        """ADC inner product should match direct computation."""
        q = ProductQuantizer(d=D, num_bits=2, seed=METHOD_SEED)
        q.fit(self.X_train)
        x = self.X_test[0]
        query = self.X_test[1]
        qv = q.quantize(x)
        x_hat = q.dequantize(qv)
        adc_ip = q.ip_adc(query, qv)
        direct_ip = np.dot(query, x_hat)
        np.testing.assert_allclose(adc_ip, direct_ip, atol=1e-10)

    def test_storage_bits(self):
        q = ProductQuantizer(d=D, num_bits=2, seed=METHOD_SEED)
        q.fit(self.X_train)
        qv = q.quantize(self.X_test[0])
        bits = q.storage_bits(qv)
        assert bits > 0
        assert bits < 10 * D  # sanity: less than 10 bits per dim


class TestOPQ:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X_train = random_unit_vectors(N_TRAIN, D, seed=DATA_SEED)
        self.X_test = random_unit_vectors(N_TEST, D, seed=DATA_SEED + 10)

    def test_opq_beats_or_matches_pq(self):
        """OPQ should achieve ≤ PQ MSE (learned rotation helps)."""
        pq = ProductQuantizer(d=D, num_bits=2, seed=METHOD_SEED)
        pq.fit(self.X_train)

        opq = OptimizedPQ(d=D, num_bits=2, seed=METHOD_SEED)
        opq.fit(self.X_train, n_iter=5)

        X_hat_pq = pq.dequantize_batch(pq.quantize_batch(self.X_test))
        X_hat_opq = opq.dequantize_batch(opq.quantize_batch(self.X_test))

        mse_pq = mse_distortion(self.X_test, X_hat_pq)
        mse_opq = mse_distortion(self.X_test, X_hat_opq)

        # OPQ should be equal or better (with small margin for noise)
        assert mse_opq <= mse_pq + 0.02, f"OPQ MSE={mse_opq:.4f} worse than PQ MSE={mse_pq:.4f}"

    def test_rotation_is_orthogonal(self):
        """Learned rotation R should be orthogonal."""
        opq = OptimizedPQ(d=D, num_bits=2, seed=METHOD_SEED)
        opq.fit(self.X_train, n_iter=3)
        R = opq._R
        np.testing.assert_allclose(R @ R.T, np.eye(D), atol=1e-10)

    def test_untrained_raises(self):
        opq = OptimizedPQ(d=D, num_bits=2, seed=METHOD_SEED)
        with pytest.raises(RuntimeError, match="trained"):
            opq.quantize(self.X_test[0])
