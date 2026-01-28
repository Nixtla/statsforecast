"""Tests for Issue #1032: forecast_fitted_values() h>1 validation."""

import numpy as np
import pandas as pd
import pytest

from statsforecast.core import StatsForecast
from statsforecast.models import Naive


@pytest.fixture
def simple_series():
    """Create a simple time series for testing."""
    return pd.DataFrame(
        {
            "unique_id": ["series_1"] * 20,
            "ds": pd.date_range("2020-01-01", periods=20, freq="D"),
            "y": np.arange(1, 21, dtype=float),
        }
    )


class TestForecastFittedValuesHorizon:
    """Test forecast_fitted_values h parameter validation."""

    def test_forecast_fitted_values_h1_works(self, simple_series):
        """Test that h=1 (default) works correctly."""
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.forecast(df=simple_series, h=5, fitted=True)

        # h=1 should work (default)
        fitted = sf.forecast_fitted_values()
        assert fitted is not None
        assert len(fitted) == len(simple_series)

        # Explicit h=1 should also work
        fitted_explicit = sf.forecast_fitted_values(h=1)
        pd.testing.assert_frame_equal(fitted, fitted_explicit)

    def test_forecast_fitted_values_h_greater_than_1_raises(self, simple_series):
        """Test that h>1 raises ValueError with helpful message."""
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.forecast(df=simple_series, h=5, fitted=True)

        # h=2 should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            sf.forecast_fitted_values(h=2)

        error_msg = str(exc_info.value)
        assert "h=1" in error_msg
        assert "cross_validation" in error_msg or "forecast()" in error_msg

    def test_forecast_fitted_values_h5_raises(self, simple_series):
        """Test that h=5 also raises ValueError."""
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.forecast(df=simple_series, h=5, fitted=True)

        with pytest.raises(ValueError):
            sf.forecast_fitted_values(h=5)

    def test_forecast_fitted_values_without_fitted_raises(self, simple_series):
        """Test that calling without fitted=True still raises appropriate error."""
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.forecast(df=simple_series, h=5, fitted=False)

        with pytest.raises(Exception) as exc_info:
            sf.forecast_fitted_values()

        assert "fitted=True" in str(exc_info.value)
