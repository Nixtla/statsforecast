"""Tests for the generate() method on model classes.

Tests sample trajectory generation functionality added in response to
GitHub Issue #1023: Support for generating sample trajectories.
"""

import numpy as np
import pytest

from statsforecast.ets import ets_f, generate_ets_samples
from statsforecast.models import AutoETS


class TestGenerateEtsSamples:
    """Tests for the low-level generate_ets_samples helper function."""

    def test_ann_model_basic(self):
        """Test generate_ets_samples with simple ANN (no trend, no season) model."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)

        model = ets_f(y, m=1)
        assert model["components"][:3] == "ANN", "Expected ANN model"

        samples = generate_ets_samples(model, h=12, n_samples=100, random_state=42)

        assert samples.shape == (100, 12)
        assert not np.any(np.isnan(samples))
        # Mean should be close to last level
        assert abs(samples.mean() - model["states"][-1, 0]) < 10

    def test_aaa_model_basic(self):
        """Test generate_ets_samples with AAA (trend + season) model."""
        np.random.seed(42)
        t = np.arange(120)
        y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 2
        y = y.astype(np.float64)

        model = ets_f(y, m=12)
        assert model["components"][:3] == "AAA", "Expected AAA model"

        samples = generate_ets_samples(model, h=12, n_samples=100, random_state=42)

        assert samples.shape == (100, 12)
        assert not np.any(np.isnan(samples))

    def test_reproducibility(self):
        """Test that random_state makes results reproducible."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)
        model = ets_f(y, m=1)

        samples1 = generate_ets_samples(model, h=12, n_samples=50, random_state=123)
        samples2 = generate_ets_samples(model, h=12, n_samples=50, random_state=123)

        np.testing.assert_allclose(samples1, samples2)

    def test_different_seeds_differ(self):
        """Test that different random seeds produce different results."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)
        model = ets_f(y, m=1)

        samples1 = generate_ets_samples(model, h=12, n_samples=50, random_state=123)
        samples2 = generate_ets_samples(model, h=12, n_samples=50, random_state=456)

        assert not np.allclose(samples1, samples2)

    def test_bootstrap_mode(self):
        """Test bootstrap mode resamples from residuals."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)
        model = ets_f(y, m=1)

        samples_parametric = generate_ets_samples(
            model, h=12, n_samples=100, bootstrap=False, random_state=42
        )
        samples_bootstrap = generate_ets_samples(
            model, h=12, n_samples=100, bootstrap=True, random_state=42
        )

        # Both should have valid shapes
        assert samples_parametric.shape == (100, 12)
        assert samples_bootstrap.shape == (100, 12)

        # They should be different (different sampling methods)
        assert not np.allclose(samples_parametric, samples_bootstrap)

    def test_n_samples_parameter(self):
        """Test that n_samples controls the number of trajectories."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)
        model = ets_f(y, m=1)

        for n in [10, 50, 200]:
            samples = generate_ets_samples(model, h=12, n_samples=n, random_state=42)
            assert samples.shape[0] == n

    def test_horizon_parameter(self):
        """Test that h controls the forecast horizon."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)
        model = ets_f(y, m=1)

        for h in [1, 6, 24]:
            samples = generate_ets_samples(model, h=h, n_samples=50, random_state=42)
            assert samples.shape[1] == h


class TestAutoETSGenerate:
    """Tests for the AutoETS.generate() method."""

    def test_generate_basic(self):
        """Test basic generate() functionality."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)

        model = AutoETS(season_length=1)
        model.fit(y)

        samples = model.generate(h=12, n_samples=100, random_state=42)

        assert samples.shape == (100, 12)
        assert not np.any(np.isnan(samples))

    def test_generate_seasonal(self):
        """Test generate() with seasonal model."""
        np.random.seed(42)
        t = np.arange(120)
        y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 2
        y = y.astype(np.float64)

        model = AutoETS(season_length=12)
        model.fit(y)

        samples = model.generate(h=12, n_samples=100, random_state=42)

        assert samples.shape == (100, 12)
        assert not np.any(np.isnan(samples))

    def test_generate_not_fitted_raises(self):
        """Test that generate() raises error if model not fitted."""
        model = AutoETS(season_length=1)

        with pytest.raises(Exception, match="fit"):
            model.generate(h=12)

    def test_generate_reproducibility(self):
        """Test that random_state makes generate() reproducible."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)

        model = AutoETS(season_length=1)
        model.fit(y)

        samples1 = model.generate(h=12, n_samples=50, random_state=123)
        samples2 = model.generate(h=12, n_samples=50, random_state=123)

        np.testing.assert_allclose(samples1, samples2)

    def test_generate_bootstrap(self):
        """Test generate() with bootstrap=True."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)

        model = AutoETS(season_length=1)
        model.fit(y)

        samples = model.generate(h=12, n_samples=100, bootstrap=True, random_state=42)

        assert samples.shape == (100, 12)
        assert not np.any(np.isnan(samples))

    def test_generate_default_parameters(self):
        """Test generate() with default parameters."""
        np.random.seed(42)
        y = 100 + np.random.randn(50) * 5
        y = y.astype(np.float64)

        model = AutoETS(season_length=1)
        model.fit(y)

        # Should work with just horizon specified
        samples = model.generate(h=12)

        assert samples.shape == (100, 12)  # default n_samples=100
        assert not np.any(np.isnan(samples))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
