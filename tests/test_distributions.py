import numpy as np
import pytest
from scipy import stats

from statsforecast import distributions as D


def test_valid_distributions_matches_utils():
    from statsforecast.utils import _VALID_DISTRIBUTIONS
    assert D.VALID_DISTRIBUTIONS == _VALID_DISTRIBUTIONS
    assert set(str(d) for d in D.VALID_DISTRIBUTIONS) == {
        "normal", "laplace", "t", "skew-normal", "ged"
    }


def test_n_extra_params():
    assert D.distribution_n_extra_params("laplace") == 0
    assert D.distribution_n_extra_params("normal") == 0
    for d in ("t", "skew-normal", "ged"):
        assert D.distribution_n_extra_params(d) == 2


def test_dist_init_params_layout():
    var = 4.0
    assert D.dist_init_params("laplace", var) == (0, [])
    n_t, init_t = D.dist_init_params("t", var)
    assert n_t == 2
    assert init_t == [np.log(var), np.log(3.0)]
    n_sn, init_sn = D.dist_init_params("skew-normal", var)
    assert init_sn == [np.log(var), 0.0]
    n_ged, init_ged = D.dist_init_params("ged", var)
    assert init_ged == [0.5 * np.log(var), np.log(2.0)]


def test_extract_dist_params():
    # t: tail = [log_sigma2, log(nu-2)]
    out = D.extract_dist_params("t", np.array([np.log(9.0), np.log(3.0)]))
    assert out["nu"] == pytest.approx(5.0)
    assert out["sigma2"] == pytest.approx(9.0)
    # skew-normal: tail = [log_sigma2, alpha]
    out = D.extract_dist_params("skew-normal", np.array([np.log(9.0), 1.5]))
    assert out["alpha_dist"] == pytest.approx(1.5)
    assert out["sigma2"] == pytest.approx(9.0)
    # ged: tail = [log_sigma, log_beta], sigma2 = exp(x)**2
    out = D.extract_dist_params("ged", np.array([np.log(3.0), np.log(2.0)]))
    assert out["beta_dist"] == pytest.approx(2.0)
    assert out["sigma2"] == pytest.approx(9.0)
    # laplace: sigma2 from residuals, no shape key
    out = D.extract_dist_params("laplace", np.array([]),
                                residuals=np.array([-1.0, 1.0, -1.0, 1.0]))
    assert out["sigma2"] == pytest.approx(2.0)  # 2 * b_hat**2, b_hat = 1


def test_quantiles_match_scipy():
    level = np.array([80, 95])
    p = 0.5 + level / 200
    np.testing.assert_allclose(D._quantiles(level, "t", {"nu": 5.0}),
                               stats.t.ppf(p, df=5.0))
    np.testing.assert_allclose(D._quantiles(level, "laplace"),
                               stats.laplace.ppf(p))
    np.testing.assert_allclose(D._quantiles(level, "skew-normal", {"alpha_dist": 1.5}),
                               stats.skewnorm.ppf(p, a=1.5))
    np.testing.assert_allclose(D._quantiles(level, "ged", {"beta_dist": 1.2}),
                               stats.gennorm.ppf(p, beta=1.2))


def test_switch_distribution_maps_all():
    from statsforecast._lib import ets as _ets
    assert D.switch_distribution("t", _ets) == _ets.Distribution.StudentT
    assert D.switch_distribution("ged", _ets) == _ets.Distribution.GED
    with pytest.raises(ValueError):
        D.switch_distribution("cauchy", _ets)


from statsforecast.utils import AirPassengers as ap


@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_ces_distribution_keys(distribution):
    from statsforecast.ces import cesmodel
    m = cesmodel(ap, m=12, seasontype="S", alpha_0=None, alpha_1=None,
                 beta_0=None, beta_1=None, nmse=3, distribution=distribution)
    assert m["distribution"] == distribution
    assert np.isfinite(m["loglik"]) and np.isfinite(m["aic"])
    assert ("nu" in m) == (distribution == "t")
    assert ("alpha_dist" in m) == (distribution == "skew-normal")
    assert ("beta_dist" in m) == (distribution == "ged")


@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_ces_interval_ordering(distribution):
    from statsforecast.models import AutoCES
    model = AutoCES(season_length=12, model="S", distribution=distribution)
    model.fit(ap)
    assert model.model_["distribution"] == distribution
    pred = model.predict(h=12, level=[80, 95])
    assert np.all(pred["lo-95"] < pred["lo-80"])
    assert np.all(pred["lo-80"] < pred["mean"])
    assert np.all(pred["mean"] < pred["hi-80"])
    assert np.all(pred["hi-80"] < pred["hi-95"])


@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_theta_distribution_keys(distribution):
    from statsforecast.theta import thetamodel
    m = thetamodel(ap, m=12, modeltype="STM", initial_smoothed=ap[0] / 2,
                   alpha=0.5, theta=2.0, nmse=3, distribution=distribution)
    assert m["distribution"] == distribution
    if distribution != "normal":
        assert np.isfinite(m["loglik"]) and np.isfinite(m["aic"])
    assert ("nu" in m) == (distribution == "t")
    assert ("alpha_dist" in m) == (distribution == "skew-normal")
    assert ("beta_dist" in m) == (distribution == "ged")


@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_theta_interval_ordering(distribution):
    from statsforecast.models import AutoTheta
    model = AutoTheta(season_length=12, model="STM", distribution=distribution)
    model.fit(ap)
    assert model.model_["distribution"] == distribution
    pred = model.predict(h=12, level=[80, 95])
    assert np.all(pred["lo-95"] < pred["lo-80"])
    assert np.all(pred["lo-80"] < pred["mean"])
    assert np.all(pred["mean"] < pred["hi-80"])
    assert np.all(pred["hi-80"] < pred["hi-95"])


import warnings


def test_simulate_distribution_mismatch_warns():
    from statsforecast.models import AutoETS
    model = AutoETS(season_length=12, model="ANN", distribution="t")
    model.fit(ap)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.simulate(h=4, n_paths=10, error_distribution="laplace")
        assert any("distribution" in str(wi.message).lower() for wi in w)


def test_simulate_defaults_to_fitted_distribution():
    from statsforecast.models import AutoETS
    model = AutoETS(season_length=12, model="ANN", distribution="t")
    model.fit(ap)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.simulate(h=4, n_paths=10)  # no explicit error_distribution
        assert not any("distribution" in str(wi.message).lower() for wi in w)


# ---------------------------------------------------------------------------
# Task 8: Cross-method distribution key-contract matrix
# ---------------------------------------------------------------------------

NON_NORMAL = ["laplace", "t", "skew-normal", "ged"]

# Each entry: (model_name, module_path, class_name, extra_kwargs)
# AutoARIMA needs method="CSS-ML" for non-normal distributions (ML optimizer required).
_MATRIX_CASES = [
    ("AutoARIMA", "statsforecast.models", "AutoARIMA",
     dict(season_length=12, method="CSS-ML")),
    ("AutoETS", "statsforecast.models", "AutoETS",
     dict(season_length=12, model="ANN")),
    ("AutoCES", "statsforecast.models", "AutoCES",
     dict(season_length=12, model="S")),
    ("AutoTheta", "statsforecast.models", "AutoTheta",
     dict(season_length=12, model="STM")),
]

_MATRIX_PARAMS = [
    pytest.param(model_name, mod, cls, extra, dist, id=f"{model_name}-{dist}")
    for (model_name, mod, cls, extra) in _MATRIX_CASES
    for dist in (["normal"] + NON_NORMAL)
]


@pytest.mark.parametrize(
    "model_name,mod_name,cls_name,extra_kwargs,distribution",
    _MATRIX_PARAMS,
)
def test_cross_method_distribution_keys(
    model_name, mod_name, cls_name, extra_kwargs, distribution
):
    import importlib

    Model = getattr(importlib.import_module(mod_name), cls_name)
    model = Model(**extra_kwargs, distribution=distribution)
    model.fit(ap)
    md = model.model_

    # `distribution` key must be present and correct
    assert md.get("distribution", "normal") == distribution

    # Shape-parameter key contract
    assert ("nu" in md) == (distribution == "t")
    assert ("alpha_dist" in md) == (distribution == "skew-normal")
    assert ("beta_dist" in md) == (distribution == "ged")

    # For non-normal distributions all models must expose sigma2
    if distribution != "normal":
        assert md.get("sigma2") is not None, (
            f"{model_name}/{distribution}: sigma2 missing from model dict"
        )


# ---------------------------------------------------------------------------
# Task 8: Heavy-tailed AIC sanity test
# ---------------------------------------------------------------------------


def test_ets_t_aic_better_than_normal_heavy_tails():
    """On Student-t AR(1) data, ETS t-distribution AIC < normal AIC."""
    from scipy import stats as scipy_stats
    from statsforecast.ets import ets_f

    rng = np.random.default_rng(42)
    n = 300
    e = scipy_stats.t.rvs(df=5, size=n, random_state=rng)
    y = np.zeros(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + e[i]
    y = y - y.min() + 1.0  # make strictly positive for ETS

    m_normal = ets_f(y, m=1, model="ANN", distribution="normal")
    m_t = ets_f(y, m=1, model="ANN", distribution="t")
    assert m_t["aic"] < m_normal["aic"], (
        f"Expected t-AIC ({m_t['aic']:.2f}) < normal-AIC ({m_normal['aic']:.2f})"
        " on heavy-tailed data"
    )
