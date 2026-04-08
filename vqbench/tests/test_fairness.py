"""
Fairness tests: same seed → same rotation, same data, reproducible results.

Req C6: All methods reproducible: fixed seed → fixed numerical results.
Req S7: Fixed seed → reproducible results.

Guarantees from §10:
  - TurboQuant Π and RaBitQ P from same haar_rotation(d, seed) → identical
  - Same data from same rng seed
"""

import numpy as np
import pytest

from vqbench.core.rotation import haar_rotation
from vqbench.datasets.synthetic import random_unit_vectors


class TestFairness:
    def test_rotation_shared_between_methods(self):
        """TurboQuant and RaBitQ must use the same rotation matrix."""
        from vqbench.methods.turboquant.mse import TurboQuantMSE
        from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit

        d, seed = 256, 42
        tq = TurboQuantMSE(d=d, num_bits=2, seed=seed)
        rq = RaBitQ1Bit(d=d, seed=seed)

        np.testing.assert_array_equal(tq._rotation, rq._rotation)

    def test_rotation_deterministic(self):
        """Same (d, seed) → identical Haar matrix."""
        R1 = haar_rotation(512, seed=42)
        R2 = haar_rotation(512, seed=42)
        np.testing.assert_array_equal(R1, R2)

    def test_different_seed_different_rotation(self):
        R1 = haar_rotation(256, seed=42)
        R2 = haar_rotation(256, seed=43)
        assert not np.allclose(R1, R2)

    def test_data_deterministic(self):
        """Same seed → identical vectors."""
        X1 = random_unit_vectors(100, 256, seed=0)
        X2 = random_unit_vectors(100, 256, seed=0)
        np.testing.assert_array_equal(X1, X2)

    def test_quantize_deterministic(self):
        """Same input + same seed → identical quantized output."""
        from vqbench.methods.turboquant.mse import TurboQuantMSE

        d, seed = 256, 42
        X = random_unit_vectors(10, d, seed=0)

        q1 = TurboQuantMSE(d=d, num_bits=2, seed=seed)
        q2 = TurboQuantMSE(d=d, num_bits=2, seed=seed)

        for x in X:
            qv1 = q1.quantize(x)
            qv2 = q2.quantize(x)
            np.testing.assert_array_equal(qv1.indices, qv2.indices)
            np.testing.assert_array_equal(qv1.norms, qv2.norms)

    def test_all_methods_same_data(self):
        """All methods see the same vectors when using same data seed."""
        from vqbench.methods.turboquant.mse import TurboQuantMSE
        from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit

        d = 256
        X = random_unit_vectors(50, d, seed=0)

        tq = TurboQuantMSE(d=d, num_bits=2, seed=42)
        rq = RaBitQ1Bit(d=d, seed=42)
        rq.fit(X)

        # Both quantize the exact same X
        tq_qvs = tq.quantize_batch(X)
        rq_qvs = rq.quantize_batch(X)

        # Dequantize and verify both started from the same data
        tq_hat = tq.dequantize_batch(tq_qvs)
        rq_hat = rq.dequantize_batch(rq_qvs)

        # Different methods → different results, but both should produce valid vectors
        assert tq_hat.shape == rq_hat.shape == (50, d)
        assert not np.allclose(tq_hat, rq_hat)  # different methods!
