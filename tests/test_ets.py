import numpy as np
import pytest
from statsforecast.ets import (
    _class3models,
    ets_f,
    etssimulate,
    forecast_ets,
    forward_ets,
    switch,
)
from statsforecast.utils import AirPassengers as ap


@pytest.fixture
def intermitent_series():
    intermitent_series = np.array([
    1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 3., 0., 1., 0., 0., 0., 0.,
    0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 3., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1., 1.,
    0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
    0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 3., 1., 0., 1., 0., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2.,
    1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1., 2., 0.,
    1., 0., 2., 2., 0., 0., 1., 2., 0., 0., 0., 2., 0., 1., 0., 0., 0.,
    0., 2., 0., 1., 0., 2., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 1., 1., 0.,
    0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 2.,
    1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 2., 0., 1., 0.,
    0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0.,
    1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 2., 0., 0., 0., 0., 1., 0., 1., 0., 2., 0., 0., 2., 0., 0.,
    2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 2.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
    1., 0., 1., 3., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 2., 0., 0., 1., 0., 2., 0., 0., 0., 0.,
    2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
    1., 0., 1., 0., 0., 0., 0., 3., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
    0., 0., 0., 2., 0., 1., 0., 2., 1., 2., 2., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 2., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 2., 2.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 4., 0., 0., 0., 0., 0., 1.,
    1., 0., 0., 1., 1., 0., 0., 2., 1., 1., 1., 2., 1., 0., 0., 0., 1.,
    0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 1., 2., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
    1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
    1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0.,
    1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
    0., 0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1.,
    0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
    1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1.,
    0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
    0., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)  # fmt: skip
    return intermitent_series


def test_forward_ets(intermitent_series):
    res = ets_f(ap, m=12)
    assert (
        forecast_ets(forward_ets(res, ap), h=12)["mean"].all()
        == forecast_ets(res, h=12)["mean"].all()
    )
    assert forward_ets(res, ap)["sigma2"].all() == res["sigma2"].all()
    assert (
        forecast_ets(forward_ets(res, ap), h=12, level=[80, 90])["lo-80"].all()
        == forecast_ets(res, h=12, level=[80, 90])["lo-80"].all()
    )
    # test tranfer
    forecast_ets(forward_ets(res, intermitent_series), h=12, level=[80, 90])
    res_transfer = forward_ets(res, intermitent_series)
    np.testing.assert_array_equal(res["par"], res_transfer["par"])


@pytest.mark.parametrize("error", ["M", "A", "Z"])
@pytest.mark.parametrize("trend", ["N", "M", "A", "Z"])
@pytest.mark.parametrize("seasonal", ["N", "M", "A", "Z"])
def test_all_model_combinations(error, trend, seasonal):
    # test holt winters
    np.random.seed(123)
    seasonal_data = 0.5 + np.random.randn(156)
    seasonal_data = np.cumsum(seasonal_data)
    mod = ets_f(seasonal_data, m=52, model="AAA")
    assert not np.isnan(forecast_ets(mod, 2)["mean"]).any()
    # test holt winters
    # Generate simulated sales data for 3 years (156 weeks)
    np.random.seed(123)
    sales = 10 + np.random.randn(156) + 0.05 * np.arange(156)
    # sales = np.cumsum(sales)

    # Add seasonality pattern for July
    for i in range(3):
        sales[i * 52 + 48] += 20 + i * 15
        sales[i * 52 + 47] += 5 + i * 6.15

    # Split the data into training and testing sets
    train_size = int(0.9 * len(sales))
    train_data = sales[:train_size]
    test_data = sales[train_size:]

    mod = ets_f(train_data, m=52, model="ZAZ")
    assert not np.isnan(forecast_ets(mod, len(test_data), level=[80])["mean"]).any()

    # test all model combinations
    # for error in ["M", "A", "Z"]:
    #     for trend in ["N", "M", "A", "Z"]:
    #         for seasonal in ["N", "M", "A", "Z"]:
    model = f"{error}{trend}{seasonal}"
    mod = ets_f(train_data, m=52, model=model, restrict=False)
    forecasts = forecast_ets(mod, len(test_data), level=[80])
    mape = np.abs(forecasts["mean"] / test_data - 1).mean()
    assert mape < 0.3


# ---- Distribution tests ----

@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_distribution_model_dict_keys(distribution):
    """'distribution' always in dict; nu/alpha_dist/beta_dist only when appropriate."""
    m = ets_f(ap, m=12, model="ANN", distribution=distribution)
    assert m["distribution"] == distribution
    if distribution == "t":
        assert "nu" in m
        assert "alpha_dist" not in m
        assert "beta_dist" not in m
    elif distribution == "skew-normal":
        assert "alpha_dist" in m
        assert "nu" not in m
        assert "beta_dist" not in m
    elif distribution == "ged":
        assert "beta_dist" in m
        assert "nu" not in m
        assert "alpha_dist" not in m
    else:
        assert "nu" not in m
        assert "alpha_dist" not in m
        assert "beta_dist" not in m


def test_distribution_invalid_raises():
    """Unknown distribution string must raise ValueError."""
    with pytest.raises(ValueError, match="distribution must be one of"):
        ets_f(ap, m=12, distribution="cauchy")


def test_t_aic_better_than_normal_heavy_tails():
    """On heavy-tailed data, t-distribution ETS AIC < normal ETS AIC."""
    rng = np.random.default_rng(42)
    n = 300
    # AR(1) level with Student-t innovations (nu=5)
    from scipy.stats import t as t_dist
    e = t_dist.rvs(df=5, size=n, random_state=rng)
    y = np.zeros(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + e[i]
    y = y - y.min() + 1.0  # make positive for ETS

    m_normal = ets_f(y, m=1, model="ANN", distribution="normal")
    m_t = ets_f(y, m=1, model="ANN", distribution="t")
    assert m_t["aic"] < m_normal["aic"]


def test_normal_data_ged_beta_near_two():
    """On Gaussian data, GED beta_dist should be near 2 (GED → normal)."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500) + 10.0
    m = ets_f(y, m=1, model="ANN", distribution="ged")
    # beta=2 is Gaussian; allow generous tolerance due to optimizer
    assert 0.5 < m["beta_dist"] < 5.0


def test_distribution_prediction_interval_ordering():
    """lo-95 < lo-80 < mean < hi-80 < hi-95 for all distributions."""
    for dist in ["normal", "laplace", "t", "skew-normal", "ged"]:
        m = ets_f(ap, m=12, model="ANN", distribution=dist)
        fcst = forecast_ets(m, h=12, level=[80, 95])
        assert np.all(fcst["lo-95"] < fcst["lo-80"]), f"{dist}: lo-95 not < lo-80"
        assert np.all(fcst["lo-80"] < fcst["mean"]), f"{dist}: lo-80 not < mean"
        assert np.all(fcst["mean"] < fcst["hi-80"]), f"{dist}: mean not < hi-80"
        assert np.all(fcst["hi-80"] < fcst["hi-95"]), f"{dist}: hi-80 not < hi-95"


def test_autoets_distribution():
    """AutoETS threads distribution through fit and predict."""
    from statsforecast.models import AutoETS
    model = AutoETS(season_length=12, model="ANN", distribution="t")
    model.fit(ap)
    assert model.model_["distribution"] == "t"
    assert "nu" in model.model_
    pred = model.predict(h=12, level=[95])
    assert "lo-95" in pred and "hi-95" in pred
    assert np.all(pred["lo-95"] < pred["hi-95"])


# ---- Class 3 prediction interval tests ----
# `_class3models` computes the forecast variance for models with multiplicative
# error and multiplicative seasonality (MNM, MAM, MAdM, MMM, MMdM), and is reached
# through `_compute_pred_intervals`. Its `Mh` moment-matrix recursion must run inside
# the horizon loop; when it does not, `mu` stays frozen at its 1-step value and the
# variance no longer tracks the seasonal pattern of the point forecasts.

# (label, model string, damped)
CLASS3_SPECS = [
    ("MNM", "MNM", False),
    ("MAM", "MAM", False),
    ("MAdM", "MAM", True),
    ("MMM", "MMM", False),
    ("MMdM", "MMM", True),
]


def fit_class3(model, damped):
    """Fit a class 3 model on AirPassengers and unpack the `_class3models` arguments."""
    # restrict=False is required to fit the multiplicative trend specifications
    mod = ets_f(
        np.asarray(ap, dtype=np.float64),
        m=12,
        model=model,
        damped=damped,
        restrict=False,
    )
    _, trend, _, damped_code = mod["components"]
    alpha, beta, gamma, phi = mod["par"][:4]
    args = dict(
        sigma=mod["sigma2"],
        last_state=mod["states"][-1],
        season_length=mod["m"],
        trend=trend,
        damped=damped_code,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        phi=phi,
    )
    return mod, args


@pytest.mark.parametrize("label,model,damped", CLASS3_SPECS)
def test_class3_variance_tracks_seasonal_mean(label, model, damped):
    """Variance must scale with the squared point forecast for multiplicative errors."""
    h = 24
    mod, args = fit_class3(model, damped)
    var = _class3models(h, **args)

    assert np.all(np.isfinite(var)), f"{label}: non-finite variance"
    assert np.all(var > 0), f"{label}: non-positive variance"

    mean = forecast_ets(mod, h=h)["mean"]
    corr = np.corrcoef(var, mean**2)[0, 1]
    # With the recursion frozen outside the loop this correlation collapses to ~0.
    assert corr > 0.5, f"{label}: var/mean**2 correlation {corr:.4f} too low"


def test_class3_interval_width_tracks_forecast():
    """The user-visible interval width follows the seasonal pattern of the forecast."""
    from statsforecast.models import AutoETS

    h = 24
    model = AutoETS(season_length=12, model="MAM")
    model.fit(np.asarray(ap, dtype=np.float64))
    pred = model.predict(h=h, level=[80])

    mean, lo, hi = pred["mean"], pred["lo-80"], pred["hi-80"]
    assert np.all(lo < mean)
    assert np.all(mean < hi)

    half_width = (hi - lo) / 2
    corr = np.corrcoef(half_width, mean)[0, 1]
    assert corr > 0.5, f"interval width/mean correlation {corr:.4f} too low"


def test_class3_matches_simulation():
    """Analytic MNM variance agrees with simulated paths across the whole horizon."""
    h = 24
    nsim = 20_000
    mod, args = fit_class3("MNM", False)
    var = _class3models(h, **args)

    error, trend, seasonality, _ = mod["components"]
    # par holds nan for components the model does not use; etssimulate needs the
    # neutral values instead (a trend of 0 and an undamped phi of 1).
    beta = 0.0 if np.isnan(args["beta"]) else args["beta"]
    phi = 1.0 if np.isnan(args["phi"]) else args["phi"]

    rng = np.random.default_rng(0)
    errors = rng.standard_normal((nsim, h)) * np.sqrt(args["sigma"])
    paths = np.empty((nsim, h))
    for k in range(nsim):
        y_path = np.zeros(h)
        etssimulate(
            args["last_state"],
            args["season_length"],
            switch(error),
            switch(trend),
            switch(seasonality),
            args["alpha"],
            beta,
            args["gamma"],
            phi,
            h,
            y_path,
            errors[k],
        )
        paths[k] = y_path

    rel_err = np.max(np.abs(var / paths.var(axis=0) - 1))
    assert rel_err < 0.06, f"analytic variance deviates from simulation by {rel_err:.2%}"


# Captured from the fixed implementation on AirPassengers (m=12, h=12) and
# corroborated by the simulation and correlation tests above.
expected_class3_var = {
    "MNM": np.array([
         487.115787,  517.599801,  766.705319,  794.800167,
         871.540585, 1192.776260, 1619.868664, 1705.262837,
        1360.096714, 1102.476187,  886.853397, 1168.650677,
    ]),
    "MAM": np.array([
         697.177283,  680.731069,  893.131531,  844.463569,
         847.824675, 1095.124178, 1365.869500, 1358.976906,
        1041.877007,  815.211436,  632.588431,  802.507673,
    ]),
}  # fmt: skip


@pytest.mark.parametrize("model", ["MNM", "MAM"])
def test_class3_variance_regression(model):
    """Pin the exact variances so future changes to the recursion are caught."""
    _, args = fit_class3(model, False)
    var = _class3models(12, **args)
    np.testing.assert_allclose(var, expected_class3_var[model], rtol=1e-6)


def test_class3models_signature():
    """`_class3models` takes exactly these ten parameters, in this order.

    The single call site passes them positionally, so a signature change that is not
    mirrored there would silently shift arguments rather than raise.
    """
    _, args = fit_class3("MAM", False)
    positional = _class3models(
        12,
        args["sigma"],
        args["last_state"],
        args["season_length"],
        args["trend"],
        args["damped"],
        args["alpha"],
        args["beta"],
        args["gamma"],
        args["phi"],
    )
    np.testing.assert_array_equal(positional, _class3models(h=12, **args))
