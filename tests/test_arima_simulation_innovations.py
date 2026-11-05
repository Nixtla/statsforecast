import numpy as np
import pytest
from scipy import stats
from scipy.signal import lfilter

from statsforecast.arima import make_arima, simulate_arima
from statsforecast.models import AutoARIMA


@pytest.fixture
def white_noise_model():
    return {
        "coef": {},
        "arma": (0, 0, 0, 0, 1, 0, 0),
        "model": make_arima([], [], []),
        "sigma2": 4.0,
        "x": np.array([-1.0, 0.0, 1.0]),
        "distribution": "normal",
    }


@pytest.mark.parametrize(
    "distribution,parameters,reference",
    [
        ("normal", {}, stats.norm(scale=2)),
        ("t", {"nu": 4.0}, stats.t(df=4, scale=2)),
        ("ged", {"beta_dist": 1.0}, stats.gennorm(beta=1, scale=2)),
        ("laplace", {}, stats.laplace(scale=np.sqrt(2))),
        ("skew-normal", {"alpha_dist": 4.0}, stats.skewnorm(a=4, scale=2)),
    ],
)
def test_autoarima_simulation_uses_fitted_innovations(
    white_noise_model, distribution, parameters, reference
):
    model = AutoARIMA(distribution=distribution)
    model.model_ = {**white_noise_model, "distribution": distribution, **parameters}
    paths = model.simulate(h=1, n_paths=20_000, seed=17)
    quantiles = np.array([0.025, 0.5, 0.975])
    np.testing.assert_allclose(
        np.quantile(paths[:, 0], quantiles), reference.ppf(quantiles), atol=0.2
    )


@pytest.mark.parametrize(
    "phi,theta", [(0.0, 0.0), (0.5, 0.0), (0.0, 0.25), (0.5, 0.25)]
)
def test_arima_bootstrap_shock_propagates_through_state(white_noise_model, phi, theta):
    model = {
        **white_noise_model,
        "coef": {"ar1": phi, "ma1": theta},
        "arma": (1, 1, 0, 0, 1, 0, 0),
        "model": make_arima([phi], [theta], []),
        "residuals": np.array([-3.0, 7.0]),
    }
    paths = simulate_arima(
        model, h=3, n_paths=4, seed=17, error_distribution="bootstrap"
    )
    innovations = np.random.default_rng(17).choice([-3.0, 7.0], size=(4, 3))
    np.testing.assert_array_equal(paths, lfilter([1, theta], [1, -phi], innovations))
    np.testing.assert_array_equal(
        paths,
        simulate_arima(model, h=3, n_paths=4, seed=17, error_distribution="bootstrap"),
    )


def test_arima_distribution_override_changes_state_noise(white_noise_model):
    paths = simulate_arima(
        white_noise_model,
        h=1,
        n_paths=20_000,
        seed=17,
        error_distribution="t",
        error_params={"df": 4.0},
    )
    np.testing.assert_allclose(
        np.quantile(paths[:, 0], [0.025, 0.975]),
        stats.t(df=4, scale=2).ppf([0.025, 0.975]),
        atol=0.2,
    )


def test_autoarima_simulation_fit_preserves_distribution():
    y = np.random.default_rng(17).laplace(size=100)
    kwargs = dict(
        d=0,
        D=0,
        max_p=0,
        max_q=0,
        start_p=0,
        start_q=0,
        seasonal=False,
        approximation=False,
        distribution="laplace",
    )
    fitted = AutoARIMA(**kwargs).fit(y)
    np.testing.assert_array_equal(
        fitted.simulate(h=3, n_paths=4, seed=17),
        AutoARIMA(**kwargs).simulate(y=y, h=3, n_paths=4, seed=17),
    )


def test_autoarima_simulation_warns_on_distribution_override(white_noise_model):
    model = AutoARIMA(distribution="t")
    model.model_ = {**white_noise_model, "distribution": "t", "nu": 4.0}
    with pytest.warns(UserWarning, match="was fitted with distribution='t'"):
        paths = model.simulate(h=1, n_paths=4, seed=17, error_distribution="normal")
    np.testing.assert_array_equal(
        paths, simulate_arima(white_noise_model, h=1, n_paths=4, seed=17)
    )
