from typing import Any

import pytest
from statsforecast import StatsForecast
from statsforecast.core import ParallelBackend
from statsforecast.models import Naive
from statsforecast.utils import generate_series


@pytest.fixture
def df():
    df = generate_series(10).reset_index()
    df["unique_id"] = df["unique_id"].astype(str)
    return df


@pytest.fixture
def common_params():
    return {
        "freq": "D",
        "h": 12,
        "level": None,
        "fitted": False,
        "prediction_intervals": None,
        "id_col": "unique_id",
        "time_col": "ds",
        "target_col": "y",
    }


@pytest.fixture
def forecast_params():
    return {
        "X_df": None,
    }


@pytest.fixture
def cv_params():
    return {
        "n_windows": 1,
        "step_size": 1,
        "test_size": None,
        "input_size": None,
        "refit": True,
    }


class FailNaive:
    def __init__(self):
        self.uses_exog = False

    def forecast(self):
        pass

    def __repr__(self):
        return "Naive"


def test_parallel_backend_forecast(df, common_params, forecast_params):
    backend = ParallelBackend()
    fcst = backend.forecast(
        df=df,
        models=[Naive()],
        fallback_model=None,
        **common_params,
        **forecast_params,
    )
    fcst_stats = StatsForecast(models=[Naive()], freq="D").forecast(df=df, h=12)
    assert fcst.equals(fcst_stats)


def test_parallel_backend_cross_validation(df, common_params, cv_params):
    backend = ParallelBackend()
    fcst = backend.cross_validation(
        df=df,
        models=[Naive()],
        fallback_model=None,
        **common_params,
        **cv_params,
    )
    fcst_stats = StatsForecast(models=[Naive()], freq="D").cross_validation(df=df, h=12)
    assert fcst.equals(fcst_stats)


def test_parallel_backend_forecast_with_fallback(df, common_params, forecast_params):
    backend = ParallelBackend()
    fcst = backend.forecast(
        df=df,
        models=[FailNaive()],
        fallback_model=Naive(),
        **common_params,
        **forecast_params,
    )
    fcst_stats = StatsForecast(models=[Naive()], freq="D").forecast(df=df, h=12)
    assert fcst.equals(fcst_stats)


def test_parallel_backend_cross_validation_with_fallback(df, common_params, cv_params):
    backend = ParallelBackend()
    fcst = backend.cross_validation(
        df=df,
        models=[FailNaive()],
        fallback_model=Naive(),
        **common_params,
        **cv_params,
    )
    fcst_stats = StatsForecast(models=[Naive()], freq="D").cross_validation(df=df, h=12)
    assert fcst.equals(fcst_stats)
