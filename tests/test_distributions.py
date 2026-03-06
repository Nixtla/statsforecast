"""Integration tests for distributional forecast support."""
import numpy as np
import pandas as pd
import pytest

from statsforecast.distributions import (
    Distribution,
    Gaussian,
    NegBin,
    Poisson,
    StudentT,
    _get_distribution,
)
from statsforecast.models import (
    ARIMA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoTheta,
    Naive,
    SeasonalNaive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
N = 60  # enough data to fit seasonal models
LEVELS = [80, 95]


@pytest.fixture
def gaussian_series():
    return rng.normal(loc=100, scale=10, size=N).astype(float)


@pytest.fixture
def poisson_series():
    return rng.poisson(lam=15, size=N).astype(float)


@pytest.fixture
def negbin_series():
    # overdispersed count data (NBinom with r=5, p=0.25)
    return rng.negative_binomial(n=5, p=0.25, size=N).astype(float)


@pytest.fixture
def student_t_series():
    from scipy import stats
    return stats.t.rvs(df=5, loc=100, scale=10, size=N, random_state=42).astype(float)


# ---------------------------------------------------------------------------
# 1. Distribution classes / _get_distribution
# ---------------------------------------------------------------------------

class TestDistributionLookup:
    def test_string_aliases(self):
        assert isinstance(_get_distribution("gaussian"), Gaussian)
        assert isinstance(_get_distribution("normal"), Gaussian)
        assert isinstance(_get_distribution("poisson"), Poisson)
        assert isinstance(_get_distribution("negbin"), NegBin)
        assert isinstance(_get_distribution("negative_binomial"), NegBin)
        assert isinstance(_get_distribution("student_t"), StudentT)
        assert isinstance(_get_distribution("t"), StudentT)

    def test_instance_passthrough(self):
        dist = Poisson()
        assert _get_distribution(dist) is dist

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            _get_distribution("beta")


# ---------------------------------------------------------------------------
# 2. Backward compatibility — default gaussian == explicit gaussian
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_autoets_default_equals_gaussian(self, gaussian_series):
        m_default = AutoETS(season_length=1)
        m_gaussian = AutoETS(season_length=1, distribution="gaussian")

        m_default.fit(gaussian_series)
        m_gaussian.fit(gaussian_series)

        pred_default = m_default.predict(h=12, level=LEVELS)
        pred_gaussian = m_gaussian.predict(h=12, level=LEVELS)

        np.testing.assert_allclose(pred_default["mean"], pred_gaussian["mean"])
        np.testing.assert_allclose(pred_default["lo-80"], pred_gaussian["lo-80"])
        np.testing.assert_allclose(pred_default["hi-95"], pred_gaussian["hi-95"])

    def test_autoets_forecast_default_equals_gaussian(self, gaussian_series):
        m = AutoETS(season_length=1)
        res_default = m.forecast(gaussian_series, h=12, level=LEVELS)
        res_gaussian = AutoETS(season_length=1, distribution="gaussian").forecast(
            gaussian_series, h=12, level=LEVELS
        )
        np.testing.assert_allclose(res_default["mean"], res_gaussian["mean"])


# ---------------------------------------------------------------------------
# 3. Poisson distribution
# ---------------------------------------------------------------------------

class TestPoisson:
    def test_fitted_nonnegative(self, poisson_series):
        m = AutoETS(season_length=1, distribution="poisson")
        m.fit(poisson_series)
        fitted = m.predict_in_sample()["fitted"]
        assert np.all(fitted >= 0), "Poisson fitted values should be non-negative"

    def test_pi_floor_at_zero(self, poisson_series):
        m = AutoETS(season_length=1, distribution="poisson")
        m.fit(poisson_series)
        pred = m.predict(h=12, level=[80])
        assert np.all(pred["lo-80"] >= 0), "Poisson lower PIs should be >= 0"

    def test_negative_data_raises(self):
        y = np.array([-1.0, 2.0, 3.0, 4.0, 5.0])
        m = AutoETS(season_length=1, distribution="poisson")
        with pytest.raises(ValueError, match="non-negative"):
            m.fit(y)

    def test_forecast_returns_correct_keys(self, poisson_series):
        m = AutoETS(season_length=1, distribution="poisson")
        res = m.forecast(poisson_series, h=12, level=[80, 95])
        for key in ["mean", "lo-80", "hi-80", "lo-95", "hi-95"]:
            assert key in res, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 4. NegBin dispersion estimation
# ---------------------------------------------------------------------------

class TestNegBin:
    def test_dispersion_positive_finite(self, negbin_series):
        m = AutoETS(season_length=1, distribution="negbin")
        m.fit(negbin_series)
        r = m._aux_params_["r"]
        assert r > 0 and np.isfinite(r), f"r should be positive and finite, got {r}"

    def test_pi_floor_at_zero(self, negbin_series):
        m = AutoETS(season_length=1, distribution="negbin")
        m.fit(negbin_series)
        pred = m.predict(h=12, level=[80])
        assert np.all(pred["lo-80"] >= 0)

    def test_negative_data_raises(self):
        y = np.array([-1.0, 2.0, 3.0, 4.0, 5.0])
        m = AutoETS(season_length=1, distribution="negbin")
        with pytest.raises(ValueError, match="non-negative"):
            m.fit(y)


# ---------------------------------------------------------------------------
# 5. Student-t df estimation
# ---------------------------------------------------------------------------

class TestStudentT:
    def test_df_reasonable_range(self, student_t_series):
        m = ARIMA(order=(1, 0, 0), distribution="student_t")
        m.fit(student_t_series)
        df = m._aux_params_["df"]
        # True df is 5; estimated should be in a reasonable range
        assert 1.5 < df < 50, f"Estimated df {df} is outside reasonable range"

    def test_pi_wider_than_gaussian(self, gaussian_series):
        # Student-t PIs (with df close to 5) should be wider than Gaussian at 95%
        # Use the same sigma so only df drives the width
        from scipy import stats
        mean = np.ones(12) * 100.0
        sigma = 10.0
        gauss = Gaussian()
        student = StudentT()
        gauss_pi = gauss.predict_intervals(mean, sigma, [95])
        student_pi = student.predict_intervals(mean, sigma, [95], df=5.0)
        gauss_width = gauss_pi["hi-95"] - gauss_pi["lo-95"]
        student_width = student_pi["hi-95"] - student_pi["lo-95"]
        assert np.all(student_width > gauss_width)


# ---------------------------------------------------------------------------
# 6. Smoke tests: all 4 distributions × key models
# ---------------------------------------------------------------------------

TIER1_MODELS = [
    AutoETS(season_length=1),
    ARIMA(order=(1, 0, 0)),
    AutoARIMA(season_length=1),
    AutoTheta(season_length=1),
    AutoCES(season_length=1),
]
DISTRIBUTIONS = ["gaussian", "poisson", "negbin", "student_t"]


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
@pytest.mark.parametrize("model_proto", TIER1_MODELS, ids=lambda m: m.alias)
def test_tier1_smoke(dist, model_proto, poisson_series, gaussian_series):
    """Smoke test: fit, predict, predict_in_sample, forecast all run without error."""
    from statsforecast.models import AutoETS, ARIMA, AutoARIMA, AutoTheta, AutoCES
    import copy

    # Non-count distributions work with gaussian series, count with non-negative
    if dist in ("poisson", "negbin"):
        y = poisson_series
    else:
        y = gaussian_series

    # Clone and set distribution
    model = copy.copy(model_proto)
    model.distribution = dist

    h = 12

    # fit + predict
    model.fit(y)
    pred = model.predict(h=h, level=LEVELS)
    assert "mean" in pred
    assert pred["mean"].shape == (h,)

    if dist != "gaussian":
        # Non-gaussian should have distributional PIs
        pred_with_pi = model.predict(h=h, level=LEVELS)
        for l in LEVELS:
            assert f"lo-{l}" in pred_with_pi
            assert f"hi-{l}" in pred_with_pi
            assert np.all(pred_with_pi[f"lo-{l}"] <= pred_with_pi[f"hi-{l}"])

    # predict_in_sample
    in_sample = model.predict_in_sample(level=[80])
    assert "fitted" in in_sample

    # forecast (stateless)
    fcst = model.forecast(y=y, h=h, level=LEVELS)
    assert "mean" in fcst
    assert fcst["mean"].shape == (h,)


# ---------------------------------------------------------------------------
# 7. Naive and SeasonalNaive Tier 2 smoke test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_naive_smoke(dist, poisson_series, gaussian_series):
    y = poisson_series if dist in ("poisson", "negbin") else gaussian_series
    m = Naive(distribution=dist)
    m.fit(y)
    pred = m.predict(h=12, level=LEVELS)
    assert "mean" in pred
    if dist != "gaussian":
        assert "lo-80" in pred

    fcst = m.forecast(y=y, h=12, level=LEVELS)
    assert "mean" in fcst


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_seasonal_naive_smoke(dist, poisson_series, gaussian_series):
    y = poisson_series if dist in ("poisson", "negbin") else gaussian_series
    m = SeasonalNaive(season_length=4, distribution=dist)
    m.fit(y)
    pred = m.predict(h=12, level=LEVELS)
    assert "mean" in pred
    if dist != "gaussian":
        assert "lo-80" in pred


# ---------------------------------------------------------------------------
# 8. StatsForecast integration
# ---------------------------------------------------------------------------

def test_statsforecast_integration(poisson_series):
    from statsforecast import StatsForecast

    df = pd.DataFrame({
        "unique_id": "ts1",
        "ds": pd.date_range("2020-01-01", periods=len(poisson_series), freq="ME"),
        "y": poisson_series,
    })

    sf = StatsForecast(
        models=[AutoETS(season_length=1, distribution="poisson")],
        freq="ME",
    )
    forecasts = sf.forecast(df=df, h=12, level=[90])
    assert "AutoETS-lo-90" in forecasts.columns
    assert "AutoETS-hi-90" in forecasts.columns
    assert (forecasts["AutoETS-lo-90"] <= forecasts["AutoETS-hi-90"]).all()
