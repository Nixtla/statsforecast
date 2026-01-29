"""Tests for TimeSeriesSimulator."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Direct import from file (bypasses installed statsforecast)
_generator_path = Path(__file__).parent.parent / "python" / "statsforecast" / "synthetic" / "generator.py"
spec = importlib.util.spec_from_file_location("generator", _generator_path)
_generator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_generator_module)
TimeSeriesSimulator = _generator_module.TimeSeriesSimulator


class TestTimeSeriesSimulator:
    """Tests for TimeSeriesSimulator class."""

    def test_basic_generation(self):
        """Test basic series generation."""
        sim = TimeSeriesSimulator(length=100, seed=42)
        df = sim.simulate(n_series=2)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"unique_id", "ds", "y"}
        assert len(df) == 200  # 2 series * 100 points
        assert df["unique_id"].nunique() == 2

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        sim1 = TimeSeriesSimulator(length=50, seed=42)
        sim2 = TimeSeriesSimulator(length=50, seed=42)

        df1 = sim1.simulate(n_series=3)
        df2 = sim2.simulate(n_series=3)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        sim1 = TimeSeriesSimulator(length=50, seed=42)
        sim2 = TimeSeriesSimulator(length=50, seed=123)

        df1 = sim1.simulate()
        df2 = sim2.simulate()

        assert not df1["y"].equals(df2["y"])


class TestBuiltInDistributions:
    """Tests for built-in distributions."""

    def test_normal_distribution(self):
        """Test normal distribution."""
        sim = TimeSeriesSimulator(
            length=1000,
            distribution="normal",
            dist_params={"loc": 100, "scale": 10},
            seed=42,
        )
        df = sim.simulate()

        # Check mean and std are approximately correct
        assert abs(df["y"].mean() - 100) < 2
        assert abs(df["y"].std() - 10) < 2

    def test_poisson_distribution(self):
        """Test Poisson distribution."""
        sim = TimeSeriesSimulator(
            length=1000,
            distribution="poisson",
            dist_params={"lam": 10},
            seed=42,
        )
        df = sim.simulate()

        # Poisson mean should equal lambda
        assert abs(df["y"].mean() - 10) < 1

    def test_gamma_distribution(self):
        """Test gamma distribution."""
        sim = TimeSeriesSimulator(
            length=1000,
            distribution="gamma",
            dist_params={"shape": 2, "scale": 5},
            seed=42,
        )
        df = sim.simulate()

        # Gamma mean = shape * scale
        assert abs(df["y"].mean() - 10) < 2

    def test_uniform_distribution(self):
        """Test uniform distribution."""
        sim = TimeSeriesSimulator(
            length=1000,
            distribution="uniform",
            dist_params={"low": 0, "high": 100},
            seed=42,
        )
        df = sim.simulate()

        assert df["y"].min() >= 0
        assert df["y"].max() <= 100
        assert abs(df["y"].mean() - 50) < 5

    def test_all_distributions(self):
        """Test all built-in distributions work."""
        distributions = [
            "normal",
            "poisson",
            "exponential",
            "gamma",
            "uniform",
            "binomial",
            "lognormal",
        ]

        for dist in distributions:
            sim = TimeSeriesSimulator(length=50, distribution=dist, seed=42)
            df = sim.simulate()
            assert len(df) == 50, f"Failed for {dist}"


class TestCustomDistribution:
    """Tests for custom distribution functions."""

    def test_custom_distribution_callable(self):
        """Test custom distribution via callable."""
        def custom_dist(size, rng):
            return rng.beta(2, 5, size=size) * 100

        sim = TimeSeriesSimulator(
            length=100,
            distribution=custom_dist,
            seed=42,
        )
        df = sim.simulate()

        assert len(df) == 100
        # Beta(2,5) mean = 2/(2+5) = 0.286, scaled by 100
        assert 20 < df["y"].mean() < 40

    def test_demand_with_spikes(self):
        """Test the demand with spikes example from docstring."""
        def demand_with_spikes(size, rng):
            base_demand = rng.gamma(shape=5, scale=10, size=size)
            spike_mask = rng.random(size) < 0.05
            spike_multiplier = rng.uniform(2.5, 5.0, size=size)
            demand = base_demand.copy()
            demand[spike_mask] *= spike_multiplier[spike_mask]
            return demand

        sim = TimeSeriesSimulator(
            length=500,
            distribution=demand_with_spikes,
            seed=42,
        )
        df = sim.simulate()

        # Should have some high values from spikes
        assert df["y"].max() > df["y"].mean() * 2


class TestTrend:
    """Tests for trend components."""

    def test_linear_trend(self):
        """Test linear trend."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            trend="linear",
            trend_params={"slope": 1.0, "intercept": 0},
            seed=42,
        )
        df = sim.simulate()

        # Last values should be much higher than first
        assert df["y"].iloc[-10:].mean() > df["y"].iloc[:10].mean() + 50

    def test_exponential_trend(self):
        """Test exponential trend."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            trend="exponential",
            trend_params={"base": 1.05, "scale": 1.0},
            seed=42,
        )
        df = sim.simulate()

        # Should show growth
        assert df["y"].iloc[-1] > df["y"].iloc[0]

    def test_custom_trend(self):
        """Test custom trend function."""
        def custom_trend(t):
            return np.log1p(t) * 10

        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            trend=custom_trend,
            seed=42,
        )
        df = sim.simulate()

        assert len(df) == 100


class TestSeasonality:
    """Tests for seasonality components."""

    def test_single_seasonality(self):
        """Test single seasonality period."""
        sim = TimeSeriesSimulator(
            length=21,  # 3 weeks
            distribution="normal",
            dist_params={"loc": 100, "scale": 0.1},
            seasonality=7,
            seasonality_strength=10.0,
            seed=42,
        )
        df = sim.simulate()

        # Day 0 and day 7 should have similar seasonal effect
        assert abs(df["y"].iloc[0] - df["y"].iloc[7]) < 1

    def test_multiple_seasonality(self):
        """Test multiple seasonality periods."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 100, "scale": 0.1},
            seasonality=[7, 30],
            seasonality_strength=[5.0, 10.0],
            seed=42,
        )
        df = sim.simulate()

        assert len(df) == 100

    def test_seasonality_auto_length_adjustment(self):
        """Test that length is adjusted for seasonality."""
        sim = TimeSeriesSimulator(
            length=10,  # Too short for period=30
            seasonality=30,
            seed=42,
        )
        # Length should be auto-adjusted to 3*30=90
        df = sim.simulate()
        assert len(df) == 90


class TestNoise:
    """Tests for noise addition."""

    def test_noise_addition(self):
        """Test that noise increases variance."""
        sim_no_noise = TimeSeriesSimulator(
            length=1000,
            distribution="normal",
            dist_params={"loc": 100, "scale": 1},
            noise_std=0,
            seed=42,
        )
        sim_with_noise = TimeSeriesSimulator(
            length=1000,
            distribution="normal",
            dist_params={"loc": 100, "scale": 1},
            noise_std=10,
            seed=42,
        )

        df_no_noise = sim_no_noise.simulate()
        df_with_noise = sim_with_noise.simulate()

        # With noise should have higher variance
        assert df_with_noise["y"].std() > df_no_noise["y"].std()


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_distribution(self):
        """Test error on invalid distribution name."""
        sim = TimeSeriesSimulator(distribution="invalid", seed=42)
        with pytest.raises(ValueError, match="Unknown distribution"):
            sim.simulate()

    def test_invalid_trend(self):
        """Test error on invalid trend name."""
        sim = TimeSeriesSimulator(trend="invalid", seed=42)
        with pytest.raises(ValueError, match="Unknown trend"):
            sim.simulate()

    def test_invalid_length(self):
        """Test error on invalid length."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            TimeSeriesSimulator(length=0)

    def test_mismatched_seasonality_strength(self):
        """Test error on mismatched seasonality/strength lengths."""
        sim = TimeSeriesSimulator(
            seasonality=[7, 30],
            seasonality_strength=[1.0, 2.0, 3.0],  # Wrong length
            seed=42,
        )
        with pytest.raises(ValueError, match="seasonality_strength must"):
            sim.simulate()


class TestIntegration:
    """Integration tests with statsforecast."""

    def test_output_format_compatible(self):
        """Test output is compatible with StatsForecast."""
        sim = TimeSeriesSimulator(length=50, seed=42)
        df = sim.simulate(n_series=3)

        # Check statsforecast expected columns
        assert "unique_id" in df.columns
        assert "ds" in df.columns
        assert "y" in df.columns

        # Check types
        assert np.issubdtype(df["ds"].dtype, np.datetime64)
        assert np.issubdtype(df["y"].dtype, np.floating)
