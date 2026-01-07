from typing import Any

import pytest
from statsforecast import StatsForecast
from statsforecast.core import ParallelBackend
from statsforecast.models import Naive
from statsforecast.utils import generate_series

print("[MODULE] test_distributed_core.py loaded", flush=True)


@pytest.fixture
def df():
    print("[FIXTURE] Starting df fixture generation", flush=True)
    df = generate_series(10).reset_index()
    df["unique_id"] = df["unique_id"].astype(str)
    print("[FIXTURE] df fixture generated successfully", flush=True)
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
    print("\n[TEST] Starting test_parallel_backend_forecast", flush=True)
    print("[TEST] Creating ParallelBackend instance", flush=True)
    backend = ParallelBackend()
    print("[TEST] Calling backend.forecast", flush=True)
    fcst = backend.forecast(
        df=df,
        models=[Naive()],
        fallback_model=None,
        **common_params,
        **forecast_params,
    )
    print("[TEST] backend.forecast completed", flush=True)
    print("[TEST] Calling StatsForecast.forecast", flush=True)
    fcst_stats = StatsForecast(models=[Naive()], freq="D").forecast(df=df, h=12)
    print("[TEST] StatsForecast.forecast completed", flush=True)
    print("[TEST] Asserting results", flush=True)
    assert fcst.equals(fcst_stats)
    print("[TEST] test_parallel_backend_forecast completed successfully", flush=True)


def test_parallel_backend_cross_validation(df, common_params, cv_params):
    print("\n[TEST] Starting test_parallel_backend_cross_validation", flush=True)
    print("[TEST] Creating ParallelBackend instance", flush=True)
    backend = ParallelBackend()
    print("[TEST] Calling backend.cross_validation", flush=True)
    fcst = backend.cross_validation(
        df=df,
        models=[Naive()],
        fallback_model=None,
        **common_params,
        **cv_params,
    )
    print("[TEST] backend.cross_validation completed", flush=True)
    print("[TEST] Calling StatsForecast.cross_validation", flush=True)
    fcst_stats = StatsForecast(models=[Naive()], freq="D").cross_validation(df=df, h=12)
    print("[TEST] StatsForecast.cross_validation completed", flush=True)
    print("[TEST] Asserting results", flush=True)
    assert fcst.equals(fcst_stats)
    print("[TEST] test_parallel_backend_cross_validation completed successfully", flush=True)


def test_parallel_backend_forecast_with_fallback(df, common_params, forecast_params):
    print("\n[TEST] Starting test_parallel_backend_forecast_with_fallback", flush=True)
    print("[TEST] Creating ParallelBackend instance", flush=True)
    backend = ParallelBackend()
    print("[TEST] Calling backend.forecast with fallback", flush=True)
    fcst = backend.forecast(
        df=df,
        models=[FailNaive()],
        fallback_model=Naive(),
        **common_params,
        **forecast_params,
    )
    print("[TEST] backend.forecast with fallback completed", flush=True)
    print("[TEST] Calling StatsForecast.forecast", flush=True)
    fcst_stats = StatsForecast(models=[Naive()], freq="D").forecast(df=df, h=12)
    print("[TEST] StatsForecast.forecast completed", flush=True)
    print("[TEST] Asserting results", flush=True)
    assert fcst.equals(fcst_stats)
    print("[TEST] test_parallel_backend_forecast_with_fallback completed successfully", flush=True)


def test_parallel_backend_cross_validation_with_fallback(df, common_params, cv_params):
    print("\n[TEST] Starting test_parallel_backend_cross_validation_with_fallback", flush=True)
    print("[TEST] Creating ParallelBackend instance", flush=True)
    backend = ParallelBackend()
    print("[TEST] Calling backend.cross_validation with fallback", flush=True)
    fcst = backend.cross_validation(
        df=df,
        models=[FailNaive()],
        fallback_model=Naive(),
        **common_params,
        **cv_params,
    )
    print("[TEST] backend.cross_validation with fallback completed", flush=True)
    print("[TEST] Calling StatsForecast.cross_validation", flush=True)
    fcst_stats = StatsForecast(models=[Naive()], freq="D").cross_validation(df=df, h=12)
    print("[TEST] StatsForecast.cross_validation completed", flush=True)
    print("[TEST] Asserting results", flush=True)
    assert fcst.equals(fcst_stats)
    print("[TEST] test_parallel_backend_cross_validation_with_fallback completed successfully", flush=True)


@pytest.fixture(scope="module", autouse=True)
def module_teardown():
    """Track module teardown to detect hangs after all tests complete."""
    yield
    print("\n[MODULE] All tests completed, running module teardown", flush=True)
