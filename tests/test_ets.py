import numpy as np
import pytest
from statsforecast.ets import ets_f, forecast_ets, forward_ets
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
