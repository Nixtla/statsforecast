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
