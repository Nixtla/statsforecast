"""Tests for the UCM (Unobserved Components Model)."""

import warnings

import numpy as np
import pytest

from statsforecast.ucm import (
    _build_matrices,
    kalman_filter,
    ucm_forecast,
    ucm_model,
)

warnings.simplefilter("ignore")


@pytest.fixture
def trend_series():
    """Local-linear-trend series (no seasonality)."""
    np.random.seed(0)
    n = 120
    return 50.0 + 0.5 * np.arange(n) + np.cumsum(np.random.randn(n))


@pytest.fixture
def seasonal_series():
    """Trend + seasonal (period 12) series."""
    np.random.seed(0)
    n = 144
    t = np.arange(n)
    return (
        20.0
        + 0.3 * t
        + 10.0 * np.sin(2 * np.pi * t / 12)
        + np.random.randn(n)
    )


# ---------------------------------------------------------------------------
# Computational layer
# ---------------------------------------------------------------------------
def test_build_matrices_trend_only():
    Z, T, R = _build_matrices(season_length=1)
    assert Z.shape == (1, 2)
    assert T.shape == (2, 2)
    assert R.shape == (2, 2)
    np.testing.assert_array_equal(Z[0], np.array([1.0, 0.0]))
    # Local linear trend transition.
    np.testing.assert_array_equal(T, np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_build_matrices_seasonal():
    s = 12
    Z, T, R = _build_matrices(season_length=s)
    k = 2 + (s - 1)
    assert Z.shape == (1, k)
    assert T.shape == (k, k)
    assert R.shape == (k, 3)
    # Picks level and current seasonal effect.
    assert Z[0, 0] == 1.0
    assert Z[0, 2] == 1.0
    # Seasonal effects sum-to-zero recursion.
    np.testing.assert_array_equal(T[2, 2 : 2 + (s - 1)], np.full(s - 1, -1.0))


def test_kalman_filter_loglik_finite(trend_series):
    y = trend_series
    Z, T, R = _build_matrices(season_length=1)
    Q = np.diag([1.0, 0.1])
    H = 1.0
    k = T.shape[0]
    a0 = np.zeros(k)
    P0 = 1e6 * np.eye(k)

    loglik, a_filt, one_step = kalman_filter(y, Z, T, R, Q, H, a0, P0)

    assert np.isfinite(loglik)
    assert a_filt.shape == (k,)
    assert one_step.shape == (len(y),)
    assert np.all(np.isfinite(one_step))


def test_ucm_model_params_trend_only(trend_series):
    mod = ucm_model(trend_series, season_length=1)
    # [sigma2_eps, sigma2_eta, sigma2_zeta]
    assert len(mod["params"]) == 3
    assert np.all(mod["params"] >= 0)
    assert len(mod["fitted"]) == len(trend_series)
    assert mod["season_length"] == 1


def test_ucm_model_params_seasonal(seasonal_series):
    mod = ucm_model(seasonal_series, season_length=12)
    # adds sigma2_omega
    assert len(mod["params"]) == 4
    assert len(mod["fitted"]) == len(seasonal_series)
    assert mod["season_length"] == 12


def test_ucm_forecast_shape(trend_series):
    mod = ucm_model(trend_series, season_length=1)
    h = 10
    fcst = ucm_forecast(mod, h)

    assert isinstance(fcst, dict)
    assert "mean" in fcst
    assert "fitted" in fcst
    assert len(fcst["mean"]) == h
    assert np.all(np.isfinite(fcst["mean"]))


def test_ucm_forecast_trend_direction(trend_series):
    """An upward local linear trend should extrapolate upward."""
    mod = ucm_model(trend_series, season_length=1)
    fcst = ucm_forecast(mod, 12)
    assert fcst["mean"][-1] > fcst["mean"][0]


# ---------------------------------------------------------------------------
# Model class API
# ---------------------------------------------------------------------------
def test_ucm_import():
    from statsforecast.models import UCM

    assert UCM is not None


def test_ucm_fit_predict(trend_series):
    from statsforecast.models import UCM

    model = UCM(season_length=1)
    model.fit(trend_series)
    fcst = model.predict(h=10)

    assert "mean" in fcst
    assert len(fcst["mean"]) == 10
    assert not np.any(np.isnan(fcst["mean"]))


def test_ucm_seasonal_fit_predict(seasonal_series):
    from statsforecast.models import UCM

    model = UCM(season_length=12)
    model.fit(seasonal_series)
    fcst = model.predict(h=12)

    assert len(fcst["mean"]) == 12


def test_ucm_forecast_method(trend_series):
    from statsforecast.models import UCM

    model = UCM(season_length=1)
    res = model.forecast(y=trend_series, h=10, fitted=True)

    assert "mean" in res
    assert "fitted" in res
    assert len(res["mean"]) == 10
    assert len(res["fitted"]) == len(trend_series)


def test_ucm_predict_in_sample(trend_series):
    from statsforecast.models import UCM

    model = UCM(season_length=1)
    model.fit(trend_series)
    insample = model.predict_in_sample()

    assert "fitted" in insample
    assert len(insample["fitted"]) == len(trend_series)


def test_ucm_level_not_implemented(trend_series):
    from statsforecast.models import UCM

    model = UCM(season_length=1)
    model.fit(trend_series)
    with pytest.raises(NotImplementedError):
        model.predict(h=5, level=[95])
    with pytest.raises(NotImplementedError):
        model.forecast(y=trend_series, h=5, level=[95])


def test_ucm_alias():
    from statsforecast.models import UCM

    model = UCM(season_length=12, alias="MyUCM")
    assert str(model) == "MyUCM"
    assert model.alias == "MyUCM"


def test_ucm_new():
    from statsforecast.models import UCM

    model1 = UCM(season_length=12)
    model2 = model1.new()

    assert model2.season_length == model1.season_length
    assert model2 is not model1


# ---------------------------------------------------------------------------
# StatsForecast integration
# ---------------------------------------------------------------------------
def test_statsforecast_integration():
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import UCM

    np.random.seed(0)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    y = 50 + 0.3 * np.arange(n) + np.random.randn(n) * 2
    df = pd.DataFrame({"unique_id": ["s1"] * n, "ds": dates, "y": y})

    sf = StatsForecast(models=[UCM(season_length=12)], freq="MS", n_jobs=1)
    sf.fit(df)
    fcst = sf.predict(h=6)

    assert len(fcst) == 6
    assert "UCM" in fcst.columns


def test_statsforecast_cross_validation_fitted():
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import UCM

    np.random.seed(0)
    n = 96
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    t = np.arange(n)
    y = 30 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 12) + np.random.randn(n)
    df = pd.DataFrame({"unique_id": ["s1"] * n, "ds": dates, "y": y})

    sf = StatsForecast(models=[UCM(season_length=12)], freq="MS", n_jobs=1)
    cv = sf.cross_validation(df=df, h=6, n_windows=2, fitted=True)

    assert "UCM" in cv.columns
    # Fitted values from cross-validation should be accessible.
    fitted_cv = sf.cross_validation_fitted_values()
    assert "UCM" in fitted_cv.columns


# ---------------------------------------------------------------------------
# Numerical correctness vs statsmodels
# ---------------------------------------------------------------------------
def test_ucm_matches_statsmodels(seasonal_series):
    """Forecasts should agree with statsmodels' UnobservedComponents."""
    sm = pytest.importorskip("statsmodels.api")

    y = seasonal_series
    h = 12

    sf_mod = ucm_model(y, season_length=12)
    sf_fcst = ucm_forecast(sf_mod, h)["mean"]

    sm_mod = sm.tsa.UnobservedComponents(
        y, level="local linear trend", seasonal=12, stochastic_seasonal=True
    )
    sm_res = sm_mod.fit(method="lbfgs", disp=False)
    sm_fcst = np.asarray(sm_res.forecast(h))

    # Point forecasts should be close in scale and direction.
    np.testing.assert_allclose(sf_fcst, sm_fcst, rtol=0.15, atol=0.15 * np.std(y))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
