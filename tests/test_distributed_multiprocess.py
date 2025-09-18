import pandas as pd
import pytest
from statsforecast import StatsForecast
from statsforecast.distributed.multiprocess import MultiprocessBackend
from statsforecast.models import Naive
from statsforecast.utils import generate_series


@pytest.fixture
def df():
    df = generate_series(10).reset_index()
    df["unique_id"] = df["unique_id"].astype(str)
    return df


class FailNaive:
    def forecast(self):
        pass

    def __repr__(self):
        return "Naive"

    @property
    def uses_exog(self):
        return False


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_mp_backend_forecast_and_cross_validation(df, n_jobs):
    """Test multiprocess backend with different n_jobs values."""
    backend = MultiprocessBackend(n_jobs=n_jobs)

    # Test forecast
    fcst = backend.forecast(df=df, models=[Naive()], freq="D", h=12)
    fcst_stats = StatsForecast(models=[Naive()], freq="D").forecast(df=df, h=12)
    pd.testing.assert_frame_equal(fcst, fcst_stats)

    # Test cross validation
    fcst_cv = backend.cross_validation(df=df, models=[Naive()], freq="D", h=12)
    fcst_stats_cv = StatsForecast(models=[Naive()], freq="D").cross_validation(
        df=df, h=12
    )
    pd.testing.assert_frame_equal(fcst_cv, fcst_stats_cv)


def test_mp_backend_fallback_model(df):
    """Test multiprocess backend with fallback model."""
    backend = MultiprocessBackend(n_jobs=1)

    # Test forecast with fallback model
    fcst = backend.forecast(
        df=df, models=[FailNaive()], freq="D", fallback_model=Naive(), h=12
    )
    fcst_stats = StatsForecast(models=[Naive()], freq="D").forecast(df=df, h=12)
    pd.testing.assert_frame_equal(fcst, fcst_stats)

    # Test cross validation with fallback model
    fcst_cv = backend.cross_validation(
        df=df, models=[FailNaive()], freq="D", fallback_model=Naive(), h=12
    )
    fcst_stats_cv = StatsForecast(models=[Naive()], freq="D").cross_validation(
        df=df, h=12
    )
    pd.testing.assert_frame_equal(fcst_cv, fcst_stats_cv)


def test_mp_backend_single_job(df):
    """Test multiprocess backend with single job (n_jobs=1)."""
    test_mp_backend_forecast_and_cross_validation(df, n_jobs=1)


@pytest.mark.slow
def test_mp_backend_multiple_jobs(df):
    """Test multiprocess backend with multiple jobs (n_jobs=10)."""
    # Only run this test if we want to test with many jobs
    backend = MultiprocessBackend(n_jobs=10)

    # Test forecast
    fcst = backend.forecast(df=df, models=[Naive()], freq="D", h=12)
    fcst_stats = StatsForecast(models=[Naive()], freq="D").forecast(df=df, h=12)
    pd.testing.assert_frame_equal(fcst, fcst_stats)
