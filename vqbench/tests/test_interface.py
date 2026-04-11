"""
Interface compliance test: all 7 quantizers satisfy VectorQuantizer ABC.

Req S1: All 7 quantizers pass interface test.
"""

import numpy as np
import pytest

from vqbench.core.base import VectorQuantizer, QuantizedVector
from vqbench.datasets.synthetic import random_unit_vectors

D = 128
DATA_SEED = 0
METHOD_SEED = 42
N_TRAIN = 500


def _get_all_quantizers():
    """Instantiate all 7 quantizers for testing."""
    from vqbench.methods.turboquant.mse import TurboQuantMSE
    from vqbench.methods.turboquant.prod import TurboQuantProd
    from vqbench.methods.turboquant.qjl import QJLQuantizer
    from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
    from vqbench.methods.rabitq.rabitq_ext import ExtRaBitQ
    from vqbench.methods.pq.product_quant import ProductQuantizer
    from vqbench.methods.pq.opq import OptimizedPQ

    X_train = random_unit_vectors(N_TRAIN, D, seed=DATA_SEED + 100)

    quantizers = []
    for cls in [TurboQuantMSE, TurboQuantProd]:
        q = cls(d=D, num_bits=2, seed=METHOD_SEED)
        quantizers.append(q)

    # QJL and RaBitQ1Bit are 1-bit-only — instantiating them at num_bits=2
    # used to silently coerce to 1 bit; they now raise ValueError.
    quantizers.append(QJLQuantizer(d=D, num_bits=1, seed=METHOD_SEED))

    rq = RaBitQ1Bit(d=D, num_bits=1, seed=METHOD_SEED)
    rq.fit(X_train)
    quantizers.append(rq)

    ext = ExtRaBitQ(d=D, num_bits=2, seed=METHOD_SEED)
    ext.fit(X_train)
    quantizers.append(ext)

    pq = ProductQuantizer(d=D, num_bits=2, seed=METHOD_SEED)
    pq.fit(X_train)
    quantizers.append(pq)

    opq = OptimizedPQ(d=D, num_bits=2, seed=METHOD_SEED)
    opq.fit(X_train, n_iter=3)
    quantizers.append(opq)

    return quantizers


ALL_QUANTIZERS = _get_all_quantizers()
QUANTIZER_IDS = [q.name for q in ALL_QUANTIZERS]


@pytest.mark.parametrize("q", ALL_QUANTIZERS, ids=QUANTIZER_IDS)
class TestInterface:
    def test_is_subclass(self, q):
        """Every quantizer inherits from VectorQuantizer."""
        assert isinstance(q, VectorQuantizer)

    def test_has_name(self, q):
        """name property returns a non-empty string."""
        assert isinstance(q.name, str)
        assert len(q.name) > 0

    def test_quantize_returns_quantized_vector(self, q):
        x = np.random.RandomState(DATA_SEED).randn(D)
        x = x / np.linalg.norm(x)
        qv = q.quantize(x)
        assert isinstance(qv, QuantizedVector)
        assert isinstance(qv.indices, np.ndarray)
        assert isinstance(qv.norms, np.ndarray)

    def test_dequantize_returns_ndarray(self, q):
        x = np.random.RandomState(DATA_SEED).randn(D)
        x = x / np.linalg.norm(x)
        qv = q.quantize(x)
        x_hat = q.dequantize(qv)
        assert isinstance(x_hat, np.ndarray)
        assert x_hat.shape == (D,)

    def test_storage_bits_positive(self, q):
        x = np.random.RandomState(DATA_SEED).randn(D)
        x = x / np.linalg.norm(x)
        qv = q.quantize(x)
        bits = q.storage_bits(qv)
        assert isinstance(bits, int)
        assert bits > 0

    def test_batch_quantize(self, q):
        X = random_unit_vectors(10, D, seed=DATA_SEED)
        qvs = q.quantize_batch(X)
        assert len(qvs) == 10
        assert all(isinstance(qv, QuantizedVector) for qv in qvs)

    def test_batch_dequantize(self, q):
        X = random_unit_vectors(10, D, seed=DATA_SEED)
        qvs = q.quantize_batch(X)
        X_hat = q.dequantize_batch(qvs)
        assert X_hat.shape == (10, D)
