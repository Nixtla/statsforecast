import warnings

import numpy as np
import pytest
from statsforecast.garch import (
    garch_cons,
    garch_forecast,
    garch_loglik,
    garch_model,
    garch_sigma2,
    generate_garch_data,
)

warnings.simplefilter("ignore")


@pytest.fixture
def garch_data():
    """Generate GARCH(1,1) data for testing."""
    n = 100
    w = 0.5
    alpha = np.array([0.3])
    beta = np.array([0.4])
    return generate_garch_data(n, w, alpha, beta)


@pytest.fixture
def garch_data_2_2():
    """Generate GARCH(2,2) data for testing."""

    def _make_data(n):
        w = 0.5
        alpha = np.array([0.1, 0.2])
        beta = np.array([0.3, 0.1])
        return generate_garch_data(n, w, alpha, beta)

    return _make_data


def test_generate_garch_data(garch_data_2_2):
    garch_data = garch_data_2_2(100)
    assert len(garch_data) == 100
    assert isinstance(garch_data, np.ndarray)


def test_garch_model_1_1(garch_data_2_2):
    garch_data = garch_data_2_2(100)
    mod = garch_model(garch_data, 1, 1)

    assert "coeff" in mod
    assert len(mod["coeff"]) == 3


def test_garch_model_2_2(garch_data_2_2):
    garch_data = garch_data_2_2(100)
    mod = garch_model(garch_data, 2, 2)

    assert "coeff" in mod
    assert len(mod["coeff"]) == 5


def test_garch_sigma2(garch_data_2_2):
    garch_data = garch_data_2_2(100)
    x0 = np.array([0.5, 0.1, 0.2, 0.3, 0.1])

    sigma2 = garch_sigma2(x0, garch_data, 2, 2)

    assert len(sigma2) == len(garch_data)
    assert isinstance(sigma2, np.ndarray)


def test_garch_loglik(garch_data_2_2):
    garch_data = garch_data_2_2(100)
    x0 = np.array([0.5, 0.3, 0.4])

    loglik = garch_loglik(x0, garch_data, 1, 1)

    assert isinstance(loglik, (float, np.floating))


def test_garch_cons():
    x0 = np.array([0.5, 0.3, 0.4])

    cons_val = garch_cons(x0)

    assert isinstance(cons_val, (float, np.floating, np.ndarray))


def test_garch_forecast(garch_data_2_2):
    garch_data = garch_data_2_2(100)
    mod = garch_model(garch_data, 1, 1)
    h = 10

    fcst = garch_forecast(mod, h)

    assert isinstance(fcst, dict)
    assert "mean" in fcst
    assert "sigma2" in fcst
    assert len(fcst["mean"]) == h
    assert len(fcst["sigma2"]) == h


def test_garch_coefficients_vs_arch(garch_data_2_2):
    """Test that GARCH(1,1) coefficients match arch library results for p=q case."""
    garch_data = garch_data_2_2(1000)
    mod = garch_model(garch_data, 1, 1)
    statsforecast_coeff = mod["coeff"]

    # Expected coefficients from arch library for comparison
    arch_coeff = np.array([0.3238, 0.1929, 0.6016])

    assert len(statsforecast_coeff) == len(arch_coeff)
    np.testing.assert_allclose(
        statsforecast_coeff,
        arch_coeff,
        rtol=0.1,
        atol=0.05,
        err_msg="StatsForecast GARCH coefficients differ significantly from arch library",
    )


def test_garch_coefficients_vs_arch_p_greater_q(garch_data_2_2):
    """Test that GARCH(2,1) coefficients match arch library results for p>q case."""
    garch_data = garch_data_2_2(1000)
    mod = garch_model(garch_data, 2, 1)
    statsforecast_coeff = mod["coeff"]

    # Expected coefficients from arch library for GARCH(2,1)
    arch_coeff = np.array([0.5299, 0.0920, 0.3039, 0.2846])

    assert len(statsforecast_coeff) == len(arch_coeff)
    np.testing.assert_allclose(
        statsforecast_coeff,
        arch_coeff,
        rtol=0.1,
        atol=0.05,
        err_msg="StatsForecast GARCH(2,1) coefficients differ significantly from arch library",
    )


def test_garch_coefficients_vs_arch_p_less_q(garch_data_2_2):
    """Test that GARCH(1,2) coefficients match arch library results for p<q case."""
    garch_data = garch_data_2_2(1000)
    mod = garch_model(garch_data, 1, 2)
    statsforecast_coeff = mod["coeff"]

    # Expected coefficients from arch library for GARCH(1,2)
    arch_coeff = np.array([0.3238, 0.1930, 0.6015, 9.2320e-13])

    assert len(statsforecast_coeff) == len(arch_coeff)
    np.testing.assert_allclose(
        statsforecast_coeff,
        arch_coeff,
        rtol=0.1,
        atol=0.05,
        err_msg="StatsForecast GARCH(1,2) coefficients differ significantly from arch library",
    )


def test_garch_coefficients_vs_arch_q_zero(garch_data_2_2):
    """Test that GARCH(1,0) coefficients match arch library results for q=0 case."""
    garch_data = garch_data_2_2(1000)
    mod = garch_model(garch_data, 1, 0)
    statsforecast_coeff = np.around(mod["coeff"], 5)

    # Expected coefficients from arch library for GARCH(1,0)
    arch_coeff = np.array([1.3503, 0.1227])

    assert len(statsforecast_coeff) == len(arch_coeff)
    np.testing.assert_allclose(
        statsforecast_coeff,
        arch_coeff,
        rtol=0.1,
        atol=0.05,
        err_msg="StatsForecast GARCH(1,0) coefficients differ significantly from arch library",
    )
