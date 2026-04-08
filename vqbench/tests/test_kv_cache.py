"""
KV-cache compressor end-to-end test.

Tests:
  - Compressed attention produces reasonable output
  - Different methods for K vs V
  - Compression ratio > 1
  - Outlier-aware quantization reduces MSE
"""

import numpy as np
import pytest

from vqbench.methods.turboquant.mse import TurboQuantMSE
from vqbench.methods.turboquant.prod import TurboQuantProd
from vqbench.methods.rabitq.rabitq_1bit import RaBitQ1Bit
from vqbench.kv_cache.compressor import KVCacheCompressor
from vqbench.kv_cache.attention import compressed_attention, attention_mse
from vqbench.kv_cache.outlier import OutlierAwareQuantizer

D = 128
N_TOKENS = 200
N_QUERIES = 10
DATA_SEED = 0
METHOD_SEED = 42


class TestKVCache:
    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(DATA_SEED)
        self.keys = rng.standard_normal((N_TOKENS, D)).astype(np.float64)
        self.values = rng.standard_normal((N_TOKENS, D)).astype(np.float64)
        self.queries = rng.standard_normal((N_QUERIES, D)).astype(np.float64)

    def test_basic_compression_and_attention(self):
        """Compressed attention produces finite, shaped output."""
        key_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        val_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)

        comp = KVCacheCompressor(key_q, val_q)
        comp.compress(self.keys, self.values)

        assert comp.num_tokens == N_TOKENS
        output = compressed_attention(self.queries, comp)
        assert output.shape == (N_QUERIES, D)
        assert np.all(np.isfinite(output))

    def test_single_query(self):
        """Single query (1D) input works."""
        key_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        val_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        comp = KVCacheCompressor(key_q, val_q)
        comp.compress(self.keys, self.values)

        output = compressed_attention(self.queries[0], comp)
        assert output.shape == (D,)

    def test_mixed_methods(self):
        """Different methods for K (unbiased IP) vs V (low MSE)."""
        key_q = TurboQuantProd(d=D, num_bits=2, seed=METHOD_SEED)
        val_q = TurboQuantMSE(d=D, num_bits=3, seed=METHOD_SEED)
        comp = KVCacheCompressor(key_q, val_q)
        comp.compress(self.keys, self.values)

        output = compressed_attention(self.queries, comp)
        assert output.shape == (N_QUERIES, D)
        assert np.all(np.isfinite(output))

    def test_compression_ratio(self):
        """Compression ratio should be > 1."""
        key_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        val_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        comp = KVCacheCompressor(key_q, val_q)
        comp.compress(self.keys, self.values)

        ratio = comp.compression_ratio()
        assert ratio > 1.0, f"Compression ratio {ratio:.2f} should be > 1"

    def test_attention_mse_reasonable(self):
        """Attention output MSE should be bounded."""
        key_q = TurboQuantMSE(d=D, num_bits=3, seed=METHOD_SEED)
        val_q = TurboQuantMSE(d=D, num_bits=3, seed=METHOD_SEED)
        comp = KVCacheCompressor(key_q, val_q)
        comp.compress(self.keys, self.values)

        mse = attention_mse(self.queries, self.keys, self.values, comp)
        assert mse < 10.0, f"Attention MSE={mse:.4f} too high"

    def test_more_bits_lower_mse(self):
        """More bits → lower attention MSE."""
        mses = []
        for b in [2, 3, 4]:
            key_q = TurboQuantMSE(d=D, num_bits=b, seed=METHOD_SEED)
            val_q = TurboQuantMSE(d=D, num_bits=b, seed=METHOD_SEED)
            comp = KVCacheCompressor(key_q, val_q)
            comp.compress(self.keys, self.values)
            mse = attention_mse(self.queries, self.keys, self.values, comp)
            mses.append(mse)
        # MSE should generally decrease
        assert mses[-1] < mses[0], f"4-bit MSE={mses[-1]:.4f} not lower than 2-bit MSE={mses[0]:.4f}"

    def test_incremental_compress(self):
        """Can add tokens incrementally."""
        key_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        val_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        comp = KVCacheCompressor(key_q, val_q)

        comp.compress(self.keys[:100], self.values[:100])
        assert comp.num_tokens == 100

        comp.compress(self.keys[100:], self.values[100:])
        assert comp.num_tokens == N_TOKENS


class TestOutlier:
    def test_outlier_reduces_mse(self):
        """Outlier-aware quantizer should reduce MSE on data with outlier channels."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, D))
        # Create artificial outlier channels (5% of dims have 10x variance)
        outlier_dims = rng.choice(D, size=D // 20, replace=False)
        X[:, outlier_dims] *= 10.0

        base_q = TurboQuantMSE(d=D, num_bits=2, seed=METHOD_SEED)
        outlier_q = OutlierAwareQuantizer(base_q, outlier_fraction=0.05)
        outlier_q.detect_outliers(X)

        # Compare MSE
        from vqbench.core.metrics import mse_distortion

        X_hat_base = base_q.dequantize_batch(base_q.quantize_batch(X))
        X_hat_outlier = outlier_q.dequantize_batch([outlier_q.quantize(x) for x in X])

        mse_base = mse_distortion(X, X_hat_base)
        mse_outlier = mse_distortion(X, X_hat_outlier)

        assert mse_outlier < mse_base, (
            f"Outlier MSE={mse_outlier:.4f} not lower than base MSE={mse_base:.4f}"
        )
