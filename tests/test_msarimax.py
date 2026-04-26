import numpy as np

from statsforecast.models import AutoMSARIMAX, MSARIMAX


def _seasonal_series(n=72):
    t = np.arange(n)
    rng = np.random.default_rng(0)
    return (
        10
        + 0.05 * t
        + np.sin(2 * np.pi * t / 6)
        + 0.5 * np.sin(2 * np.pi * t / 12)
        + rng.normal(scale=0.05, size=n)
    )


def test_msarimax_multiple_seasonal_forecast():
    y = _seasonal_series()
    model = MSARIMAX(
        lags=[1, 6, 12],
        ar_order=[1, 1, 0],
        i_order=[0, 1, 1],
        ma_order=[0, 0, 0],
        maxiter=50,
    ).fit(y)

    forecast = model.predict(6, level=[80])
    fitted = model.predict_in_sample()

    assert forecast["mean"].shape == (6,)
    assert "lo-80" in forecast
    assert "hi-80" in forecast
    assert np.isfinite(forecast["mean"]).all()
    assert fitted["fitted"].shape == y.shape
    assert np.isfinite(fitted["fitted"][18:]).all()


def test_msarimax_with_exogenous_regressors():
    y = _seasonal_series()
    X = np.arange(y.size, dtype=float).reshape(-1, 1)
    X_future = np.arange(y.size, y.size + 4, dtype=float).reshape(-1, 1)

    model = MSARIMAX(
        lags=[1, 6],
        ar_order=[1, 1],
        i_order=[0, 1],
        ma_order=[0, 0],
        maxiter=50,
    ).fit(y, X=X)
    forecast = model.predict(4, X=X_future)

    assert forecast["mean"].shape == (4,)
    assert np.isfinite(forecast["mean"]).all()


def test_auto_msarimax_selects_orders():
    y = _seasonal_series(60)
    model = AutoMSARIMAX(
        lags=[1, 6],
        max_ar_order=[1, 1],
        max_i_order=[1, 1],
        max_ma_order=[0, 0],
        include_constant=False,
        maxiter=20,
    ).fit(y)

    forecast = model.predict(4)

    assert forecast["mean"].shape == (4,)
    assert np.isfinite(forecast["mean"]).all()
    assert model.selected_orders_["lags"] == [1, 6]
    assert model.model_["selection"]["tried"] > 0


def test_auto_msarimax_parallel_matches_serial():
    y = _seasonal_series(60)
    kwargs = dict(
        lags=[1, 6],
        max_ar_order=[1, 1],
        max_i_order=[1, 1],
        max_ma_order=[0, 0],
        include_constant=False,
        maxiter=20,
    )

    serial = AutoMSARIMAX(**kwargs, n_jobs=1).fit(y)
    parallel = AutoMSARIMAX(**kwargs, n_jobs=2).fit(y)

    assert parallel.selected_orders_ == serial.selected_orders_
    assert parallel.model_["selection"]["tried"] == serial.model_["selection"]["tried"]
