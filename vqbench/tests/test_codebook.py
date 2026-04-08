"""
Test Lloyd-Max codebook against paper reference values.

Paper: Zandieh et al., arXiv 2504.19874, Table 1
Centroids (×√d): b=1 → ±0.7979, b=2 → ±0.4528, ±1.5104
"""

import numpy as np
import pytest

from vqbench.methods.turboquant.codebook import (
    lloyd_max_codebook,
    verify_codebook,
    CENTROIDS_SCALED_REF,
)


@pytest.mark.parametrize("d", [128, 256, 512])
class TestCodebook:
    def test_1bit_centroids(self, d):
        """b=1 centroids ×√d ≈ ±0.7979."""
        centroids, _ = lloyd_max_codebook(1, d)
        scaled = centroids * np.sqrt(d)
        np.testing.assert_allclose(scaled, CENTROIDS_SCALED_REF[1], atol=0.01)

    def test_2bit_centroids(self, d):
        """b=2 centroids ×√d ≈ ±0.4528, ±1.5104."""
        centroids, _ = lloyd_max_codebook(2, d)
        scaled = centroids * np.sqrt(d)
        np.testing.assert_allclose(scaled, CENTROIDS_SCALED_REF[2], atol=0.01)

    def test_centroids_sorted(self, d):
        for b in [1, 2, 3, 4]:
            centroids, boundaries = lloyd_max_codebook(b, d)
            assert np.all(np.diff(centroids) > 0), f"Centroids not sorted for b={b}"
            assert np.all(np.diff(boundaries) > 0), f"Boundaries not sorted for b={b}"

    def test_verify_utility(self, d):
        assert verify_codebook(1, d)
        assert verify_codebook(2, d)

    def test_codebook_symmetry(self, d):
        """Codebook for Gaussian should be symmetric around 0."""
        for b in [1, 2, 3, 4]:
            centroids, _ = lloyd_max_codebook(b, d)
            np.testing.assert_allclose(centroids, -centroids[::-1], atol=1e-10)
