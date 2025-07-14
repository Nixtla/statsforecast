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
    n = 100
    w = 0.5
    alpha = np.array([0.1, 0.2])
    beta = np.array([0.3, 0.1])
    return generate_garch_data(n, w, alpha, beta)


def test_generate_garch_data():
    n = 100
    w = 0.5
    alpha = np.array([0.3])
    beta = np.array([0.4])

    y = generate_garch_data(n, w, alpha, beta)

    assert len(y) == n
    assert isinstance(y, np.ndarray)


def test_garch_model_1_1(garch_data):
    mod = garch_model(garch_data, 1, 1)

    assert "coeff" in mod
    assert len(mod["coeff"]) == 3


def test_garch_model_2_2(garch_data_2_2):
    mod = garch_model(garch_data_2_2, 2, 2)

    assert "coeff" in mod
    assert len(mod["coeff"]) == 5


def test_garch_sigma2(garch_data_2_2):
    x0 = np.array([0.5, 0.1, 0.2, 0.3, 0.1])

    sigma2 = garch_sigma2(x0, garch_data_2_2, 2, 2)

    assert len(sigma2) == len(garch_data_2_2)
    assert isinstance(sigma2, np.ndarray)


def test_garch_loglik(garch_data):
    x0 = np.array([0.5, 0.3, 0.4])

    loglik = garch_loglik(x0, garch_data, 1, 1)

    assert isinstance(loglik, (float, np.floating))


def test_garch_cons():
    x0 = np.array([0.5, 0.3, 0.4])

    cons_val = garch_cons(x0)

    assert isinstance(cons_val, (float, np.floating, np.ndarray))


def test_garch_forecast(garch_data):
    mod = garch_model(garch_data, 1, 1)
    h = 10

    fcst = garch_forecast(mod, h)

    assert isinstance(fcst, dict)
    assert "mean" in fcst
    assert "sigma2" in fcst
    assert len(fcst["mean"]) == h
    assert len(fcst["sigma2"]) == h
