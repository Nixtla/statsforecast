import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from scipy.stats import gennorm, skewnorm
from scipy.stats import t as t_dist

from statsforecast.arima import (
    Arima,
    ARIMA_invtrans,
    AutoARIMA,
    arima,
    arima_css,
    arima_gradtrans,
    arima_like,
    arima_string,
    arima_transpar,
    arima_undopars,
    auto_arima_f,
    fitted_arima,
    fixed_params_from_dict,
    forecast_arima,
    forward_arima,
    getQ0,
    kalman_forecast,
    make_arima,
    mstl,
    myarima,
    ndiffs,
    newmodel,
    nsdiffs,
    predict_arima,
    print_statsforecast_ARIMA,
    seas_heuristic,
)
from statsforecast.models import ARIMA
from statsforecast.utils import AirPassengers as ap

warnings.simplefilter("ignore")


def test_arima_gradtrans():
    x = np.array([0.1, 0.4, 1.0, 3.1], dtype=np.float32)
    arma = np.array([1, 0, 1])
    expected = np.diag([0.9899673, 0.8553135, 1, 1])
    np.testing.assert_allclose(arima_gradtrans(x, arma), expected)


def test_arima_undopars():
    x = np.array([0.1, 0.4, 1.0, 3.1])
    arma = np.array([1, 0, 1])
    expected = np.array([0.09966799, 0.37994896, 1.00000000, 3.10000000])
    np.testing.assert_allclose(arima_undopars(x, arma), expected)


# x = np.array([0.1, 0.4, 1.0, 3.1])
# arma = np.array([1, 0, 1])
# ARIMA_invtrans(x, arma)


def test_getQ0():
    expected_getQ0 = np.array([
        [ -3.07619732,   1.11465544,   2.11357369,   3.15204201,
            4.19013718,   5.22823588,   6.26633453,   7.30443355,
            8.34249459,   9.38458115,  10.        ],
        [  1.11465544,  -3.22931088,   1.92416552,   2.84615733,
            3.80807237,   4.76961073,   5.73115265,   6.69269418,
            7.65427405,   8.61179041,  10.        ],
        [  2.11357369,   1.92416552,  -0.37881633,   5.73654439,
            7.62116681,   9.54570541,  11.46986742,  13.39403227,
            15.31827268,  17.23450038,  20.        ],
        [  3.15204201,   2.84615733,   5.73654439,   4.39470753,
            11.47233269,  14.31920899,  17.20600158,  20.0924165 ,
            22.9789482 ,  25.85347889,  30.        ],
        [  4.19013718,   3.80807237,   7.62116681,  11.47233269,
            11.09276725,  19.13264974,  22.94178352,  26.79083216,
            30.63965504,  34.47249261,  40.        ],
        [  5.22823588,   4.76961073,   9.54570541,  14.31920899,
            19.13264974,  19.71534157,  28.71748151,  33.48887095,
            38.30036514,  43.09150596,  50.        ],
        [  6.26633453,   5.73115265,  11.46986742,  17.20600158,
            22.94178352,  28.71748151,  30.2624308 ,  40.22682604,
            45.96069867,  51.71052289,  60.        ],
        [  7.30443355,   6.69269418,  13.39403227,  20.0924165 ,
            26.79083216,  33.48887095,  40.22682604,  42.73402992,
            53.66094562,  60.32916003,  70.        ],
        [  8.34249459,   7.65427405,  15.31827268,  22.9789482 ,
            30.63965504,  38.30036514,  45.96069867,  53.66094562,
            57.13074521,  68.98805242,  80.        ],
        [  9.38458115,   8.61179041,  17.23450038,  25.85347889,
            34.47249261,  43.09150596,  51.71052289,  60.32916003,
            68.98805242,  73.38026771,  90.        ],
        [ 10.        ,  10.        ,  20.        ,  30.        ,
            40.        ,  50.        ,  60.        ,  70.        ,
            80.        ,  90.        , 100.        ]]
    )  # fmt: skip
    x = np.arange(1, 11)
    np.testing.assert_allclose(expected_getQ0, getQ0(x, x))


@pytest.fixture
def expected_arima_transpar_f():
    expected_arima_transpar_f = (
        np.array([0.5, 1.0, -0.25, 0.25, -0.25, -0.25]),
        np.array([0.5, 1.0, 0.25, 0.75, 0.25, 0.25]),
    )
    return expected_arima_transpar_f


def test_arima_transpar(expected_arima_transpar_f):
    par = np.array([1.26377432, 0.82436223, -0.51341576])
    arma = (2, 1, 0, 0, 12, 1, 1)
    expected = np.array([0.2748562, 0.6774372]), np.array([-0.5134158])
    res = arima_transpar(par, arma, True)
    for actual, exp in zip(res, expected):
        np.testing.assert_allclose(actual, exp)

    params = np.repeat(0.5, 10)
    arma = np.ones(5, dtype=np.int32) * 2
    for exp, calc in zip(
        expected_arima_transpar_f, arima_transpar(params, arma, False)
    ):
        np.testing.assert_allclose(exp, calc)


# arima_css(
#     np.arange(1, 11),
#     np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
#     expected_arima_transpar_f[0],
#     expected_arima_transpar_f[1],
# )
# phi = np.array([0.68065055, 0.24123847])
# theta = np.array([-1.09653952])
# Delta = np.array(
#     [1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -1.0]
# )
# res = make_arima(phi, theta, Delta)
# y = np.arange(10, dtype=np.float64)
# phi = np.array([0.99551517])
# theta = np.array([])
# delta = np.array([1.0])
# a = np.array([0.0, 0.0])
# P = np.array([[0.0, 0.0], [0.0, 0.0]])
# Pn = np.array(
#     [
#         [5.32878591e02, 0.00000000e00],
#         [0.00000000e00, 1.00000000e06],
#     ]
# )
# up = 0
# use_resid = True
# res = arima_like(y, phi, theta, delta, a, P, Pn, up, use_resid)
# res
def test_fixed_params_from_dict():
    assert fixed_params_from_dict(
        {
            "ar1": 3,
            "ma1": 4,
            "ma2": 6,
            "ex_1": 4,
            "sar1": 0,
            "sma1": 10,
            "sar_1": 9,
            "intercept": 8,
        },
        order=(1, 0, 2),
        seasonal={"order": (1, 0, 1), "period": 1},
        intercept=True,
        n_ex=1,
    ) == [3, 4, 6, 0, 10, 8, 4]


@pytest.fixture
def drift_xreg():
    drift = np.arange(1, ap.size + 1).reshape(-1, 1)
    xreg = np.concatenate([drift, np.sqrt(drift)], axis=1)
    return drift, xreg


@pytest.mark.parametrize("method", ["CSS", "CSS-ML"])
def test_fixed_argument(method, drift_xreg):
    drift, xreg = drift_xreg
    # test fixed argument

    # for method in ["CSS", "CSS-ML"]:
    assert (
        arima(ap, order=(2, 1, 1), fixed=[0.0, np.nan, 0.0], method=method)["coef"]
        == arima(ap, order=(2, 1, 1), fixed={"ar1": 0, "ma1": 0}, method=method)["coef"]
    )

    assert (
        arima(
            ap,
            order=(2, 1, 1),
            fixed=[0.0, np.nan, 0.0, 0.5, 0.6],
            xreg=xreg,
            method=method,
        )["coef"]
        == arima(
            ap,
            order=(2, 1, 1),
            fixed={"ar1": 0, "ma1": 0, "ex_1": 0.5, "ex_2": 0.6},
            xreg=xreg,
            method=method,
        )["coef"]
    )

    assert (
        arima(ap, order=(2, 0, 1), fixed=[0.0, np.nan, 0.0, np.nan], method=method)[
            "coef"
        ]
        == arima(ap, order=(2, 0, 1), fixed={"ar1": 0, "ma1": 0}, method=method)["coef"]
    )

    assert (
        arima(ap, order=(2, 0, 1), fixed=[0.0, np.nan, 0.0, 8], method=method)["coef"]
        == arima(
            ap,
            order=(2, 0, 1),
            fixed={"ar1": 0, "ma1": 0, "intercept": 8},
            method=method,
        )["coef"]
    )

    assert (
        arima(
            ap,
            order=(2, 0, 1),
            fixed=[0.0, np.nan, 0.0, 8, np.nan, 9],
            xreg=xreg,
            method=method,
        )["coef"]
        == arima(
            ap,
            order=(2, 0, 1),
            fixed={"ar1": 0, "ma1": 0, "intercept": 8, "ex_2": 9},
            xreg=xreg,
            method=method,
        )["coef"]
    )


# arima(
#     ap,
#     order=(2, 1, 1),
#     seasonal={"order": (0, 1, 0), "period": 12},
#     include_mean=False,
#     method="CSS-ML",
# )["coef"]
# res_s = arima(
#     ap, order=(0, 1, 0), seasonal={"order": (2, 1, 0), "period": 12}, method="CSS-ML"
# )

# ##
# res_s = arima(
#     ap, order=(0, 1, 0), seasonal={"order": (2, 1, 0), "period": 12}, method="CSS-ML"
# )
# res_s["arma"], res_s["aic"], res_s["coef"], np.sqrt(np.diag(res_s["var_coef"]))
# order = (2, 1, 1)
# seasonal = {"order": (0, 1, 0), "period": 12}

# res = arima(ap, order, seasonal, method="CSS-ML")
# ## res = arima(ap, order, seasonal, method='CSS-ML')
# res["arma"], res["aic"], res["coef"], np.sqrt(np.diag(res["var_coef"]))
# res_intercept = arima(
#     ap, (2, 0, 1), {"order": (0, 0, 0), "period": 12}, method="CSS-ML"
# )
# ## res_intercept = arima(ap, (2, 0, 1), {'order': (0, 0, 0), 'period': 12}, method='CSS-ML')
# (
#     res_intercept["arma"],
#     res_intercept["aic"],
#     res_intercept["coef"],
#     np.sqrt(np.diag(res_intercept["var_coef"])),
# )
# res_xreg = arima(
#     ap, (2, 0, 1), {"order": (0, 0, 0), "period": 12}, xreg=xreg, method="CSS-ML"
# )
# ## res_xreg = arima(ap, (2, 0, 1), {'order': (0, 0, 0), 'period': 12}, xreg=xreg, method='CSS-ML')
# (
#     res_xreg["arma"],
#     res_xreg["aic"],
#     res_xreg["coef"],
#     np.sqrt(np.diag(res_xreg["var_coef"])),
# )
# # test fixed exogenous params
# arima(ap, (1, 1, 0), xreg=xreg, fixed=[0.0, np.nan, -0.1], method="CSS-ML")
# kalman_forecast(10, *(res["model"][var] for var in ["Z", "a", "P", "T", "V", "h"]))
# kalman_forecast(
#     10, *(res_intercept["model"][var] for var in ["Z", "a", "P", "T", "V", "h"])
# )
# predict_arima(res, 10)
# predict_arima(res_intercept, 10)
# newdrift = np.arange(ap.size + 1, ap.size + 10 + 1).reshape(-1, 1)
# newxreg = np.concatenate([newdrift, np.sqrt(newdrift)], axis=1)
# predict_arima(res_xreg, 10, newxreg=newxreg)
# myarima(
#     ap,
#     order=(2, 1, 1),
#     seasonal={"order": (0, 1, 0), "period": 12},
#     constant=False,
#     ic="aicc",
#     method="CSS-ML",
# )["aic"]
# res = myarima(ap, order=(0, 1, 0), xreg=drift**2)
# res["coef"]
# ## res = search_arima(ap, period=12)
# res["arma"], res["aic"]


@pytest.mark.parametrize("method", ["CSS", "CSS-ML"])
def test_Arima_fixed_argument(method, drift_xreg):
    drift, xreg = drift_xreg
    # test fixed argument
    # for method in ["CSS", "CSS-ML"]:
    assert (
        Arima(ap, order=(2, 1, 1), fixed=[0.0, np.nan, 0.0], method=method)["coef"]
        == Arima(ap, order=(2, 1, 1), fixed={"ar1": 0, "ma1": 0}, method=method)["coef"]
    )
    assert (
        Arima(
            ap,
            order=(2, 1, 1),
            fixed=[0.0, np.nan, 0.0, 0.5, 0.6],
            xreg=xreg,
            method=method,
        )["coef"]
        == Arima(
            ap,
            order=(2, 1, 1),
            fixed={"ar1": 0, "ma1": 0, "ex_1": 0.5, "ex_2": 0.6},
            xreg=xreg,
            method=method,
        )["coef"]
    )

    assert (
        Arima(ap, order=(2, 0, 1), fixed=[0.0, np.nan, 0.0, np.nan], method=method)[
            "coef"
        ]
        == Arima(ap, order=(2, 0, 1), fixed={"ar1": 0, "ma1": 0}, method=method)["coef"]
    )

    assert (
        Arima(ap, order=(2, 0, 1), fixed=[0.0, np.nan, 0.0, 8], method=method)["coef"]
        == Arima(
            ap,
            order=(2, 0, 1),
            fixed={"ar1": 0, "ma1": 0, "intercept": 8},
            method=method,
        )["coef"]
    )

    assert (
        Arima(
            ap,
            order=(2, 0, 1),
            fixed=[0.0, np.nan, 0.0, 8, np.nan, 9],
            xreg=xreg,
            method=method,
        )["coef"]
        == Arima(
            ap,
            order=(2, 0, 1),
            fixed={"ar1": 0, "ma1": 0, "intercept": 8, "ex_2": 9},
            xreg=xreg,
            method=method,
        )["coef"]
    )

    assert (
        Arima(
            ap,
            order=(2, 0, 1),
            fixed=[0.0, np.nan, 0.0, 8, np.nan, np.nan, 9],
            include_drift=True,
            xreg=xreg,
            method=method,
        )["coef"]
        == Arima(
            ap,
            order=(2, 0, 1),
            fixed={"ar1": 0, "ma1": 0, "intercept": 8, "ex_2": 9},
            include_drift=True,
            xreg=xreg,
            method=method,
        )["coef"]
    )

    assert (
        Arima(
            ap,
            order=(2, 0, 1),
            fixed=[0.0, np.nan, 0.0, 8, 8.5, np.nan, 9],
            include_drift=True,
            xreg=xreg,
            method=method,
        )["coef"]
        == Arima(
            ap,
            order=(2, 0, 1),
            fixed={"ar1": 0, "ma1": 0, "intercept": 8, "ex_2": 9, "drift": 8.5},
            include_drift=True,
            xreg=xreg,
            method=method,
        )["coef"]
    )


@pytest.fixture
def res_Arima_s():
    return Arima(
        ap,
        order=(0, 1, 0),
        seasonal={"order": (2, 1, 0), "period": 12},
        method="CSS-ML",
    )


@pytest.fixture
def res_Arima():
    return Arima(
        ap,
        seasonal={"order": (0, 0, 0), "period": 12},
        include_drift=True,
        method="CSS-ML",
    )


@pytest.fixture
def res_Arima_ex():
    drift = np.arange(1, ap.size + 1).reshape(-1, 1)

    return Arima(
        ap,
        seasonal={"order": (0, 0, 0), "period": 12},
        include_drift=True,
        xreg=np.sqrt(drift),
        method="CSS-ML",
    )


def test_Arima_model_parameter(res_Arima_s):
    """Test that Arima models can be reconstructed from existing model parameters."""

    for key in ["residuals", "arma"]:
        np.testing.assert_array_equal(
            Arima(ap, model=res_Arima_s, method="CSS-ML")[key], res_Arima_s[key]
        )


def test_Arima_drift_and_residuals():
    """Test Arima model with drift parameter and residuals consistency."""
    res_Arima = Arima(
        ap,
        seasonal={"order": (0, 0, 0), "period": 12},
        include_drift=True,
        method="CSS-ML",
    )

    # Ensure the model was created successfully
    assert res_Arima is not None

    # Verify model properties exist
    assert "arma" in res_Arima
    assert "aic" in res_Arima
    assert "coef" in res_Arima
    assert "var_coef" in res_Arima

    # Test residuals consistency
    np.testing.assert_allclose(
        Arima(ap, model=res_Arima, method="CSS-ML")["residuals"], res_Arima["residuals"]
    )


def test_Arima_with_exogenous_variables(res_Arima_ex):
    """Test Arima model with exogenous variables and residuals consistency."""
    drift = np.arange(1, ap.size + 1).reshape(-1, 1)

    np.testing.assert_allclose(
        Arima(ap, model=res_Arima_ex, method="CSS-ML", xreg=np.sqrt(drift))[
            "residuals"
        ],
        res_Arima_ex["residuals"],
    )


def test_recursive_window_average():
    """Test recursive window average forecasting with fixed AR parameters."""
    rec_window_av = Arima(
        ap,
        order=(2, 0, 0),
        include_mean=False,
        fixed={"ar1": 0.5, "ar2": 0.5},
        method="CSS-ML",
    )
    expected_fcsts = []
    for i in range(4):
        mean = np.concatenate([ap, expected_fcsts])[-2:].mean()
        expected_fcsts.append(mean)

    np.testing.assert_array_equal(
        forecast_arima(rec_window_av, 4)["mean"], np.array(expected_fcsts)
    )


def test_forecast_arima_confidence_intervals(res_Arima_s):
    """Test forecast_arima with different confidence interval configurations."""

    # Test default forecast (no intervals)
    fcst = forecast_arima(res_Arima_s, h=12)
    assert fcst["lower"] is None
    assert fcst["upper"] is None

    # Test specific confidence levels
    fcst = forecast_arima(res_Arima_s, h=12, level=(80, 95))
    assert fcst["lower"].columns.tolist() == ["80%", "95%"]
    assert fcst["upper"].columns.tolist() == ["80%", "95%"]

    # Test fan chart
    fcst = forecast_arima(res_Arima_s, fan=True, h=10)
    assert fcst["lower"].shape[1] == 17
    assert fcst["upper"].shape[1] == 17


def test_fitted_arima_lengths(res_Arima, res_Arima_ex, res_Arima_s):
    """Test that fitted_arima returns correct lengths for different models."""

    # Test fitted lengths
    fitted_res_Arima = fitted_arima(res_Arima)
    assert len(fitted_res_Arima) == len(res_Arima["x"])

    fitted_res_Arima_ex = fitted_arima(res_Arima_ex)
    assert len(fitted_res_Arima_ex) == len(res_Arima_ex["x"])

    fitted_res_Arima_s = fitted_arima(res_Arima_s)
    assert len(fitted_res_Arima_s) == len(res_Arima_s["x"])


# mstl(x, 12)
# seas_heuristic(x, 12)


@pytest.fixture
def almost_constant_x():
    return np.hstack([np.full(42, 100), np.array([119, 525])])


def test_nsdiffs_and_newmodel(almost_constant_x):
    """Test nsdiffs function and newmodel function."""
    # Test nsdiffs with seasonal period
    assert nsdiffs(ap, period=12) >= 0

    # Test nsdiffs with almost constant data

    assert nsdiffs(almost_constant_x, period=12) == 0

    # Test ndiffs
    assert ndiffs(ap) >= 0

    # Test newmodel function
    results = np.array([[0, 0, 0, 0, 1, 0, 1]])
    assert not newmodel(*results[0], results)
    assert newmodel(0, 1, 0, 0, 1, 0, 1, results)


def test_auto_arima_simple_cases():
    """Test auto_arima_f with simple cases."""
    assert math.isclose(
        auto_arima_f(np.arange(1, 3))["coef"]["intercept"],
        1.5,
        rel_tol=1e-3,
    )


def assert_forward(fitted_model, forecasts, xreg_train=None, xreg_test=None, y=ap):
    """Helper function to test forward_arima functionality."""
    np.testing.assert_allclose(
        forecast_arima(
            model=forward_arima(fitted_model=fitted_model, y=y, xreg=xreg_train),
            h=len(forecasts),
            xreg=xreg_test,
        )["mean"],
        forecasts,
    )


def test_forward_arima_models():
    """Test forward_arima functionality with various model configurations."""
    drift = np.arange(1, ap.size + 1).reshape(-1, 1)
    newdrift = np.arange(ap.size + 1, ap.size + 10 + 1).reshape(-1, 1)

    # Simple model
    mod_simple = auto_arima_f(ap, period=12, method="CSS-ML")
    mod_simple_forecasts = forecast_arima(mod_simple, 7)["mean"]
    assert_forward(mod_simple, mod_simple_forecasts)

    # Model with single exogenous variable
    mod_x_1 = auto_arima_f(ap, period=12, method="CSS-ML", xreg=np.sqrt(drift))
    mod_x_1_forecasts = forecast_arima(mod_x_1, 7, xreg=np.sqrt(newdrift))["mean"]
    assert_forward(
        mod_x_1,
        mod_x_1_forecasts,
        xreg_train=np.sqrt(drift),
        xreg_test=np.sqrt(newdrift),
    )

    # Model with multiple exogenous variables
    mod_x_2 = auto_arima_f(
        ap, period=12, method="CSS-ML", xreg=np.hstack([np.sqrt(drift), np.log(drift)])
    )
    mod_x_2_forecasts = forecast_arima(
        mod_x_2, 7, xreg=np.hstack([np.sqrt(newdrift), np.log(newdrift)])
    )["mean"]
    assert_forward(
        mod_x_2,
        mod_x_2_forecasts,
        xreg_train=np.hstack([np.sqrt(drift), np.log(drift)]),
        xreg_test=np.hstack([np.sqrt(newdrift), np.log(newdrift)]),
    )

    # Model with trace enabled
    mod = auto_arima_f(ap, period=12, method="CSS-ML", trace=True)
    mod_forecasts = forecast_arima(mod, h=12)["mean"]
    assert_forward(mod, mod_forecasts)

    # Drift model
    drift_model = auto_arima_f(np.arange(1, 101))
    drift_forecasts = forecast_arima(drift_model, 12)["mean"]
    np.testing.assert_array_equal(drift_forecasts, np.arange(101, 101 + 12))
    assert_forward(drift_model, drift_forecasts, y=np.arange(1, 101))

    # Constant model
    constant_model = auto_arima_f(np.array([1] * 36))
    constant_model_forecasts = forecast_arima(constant_model, 12)["mean"]
    assert_forward(constant_model, constant_model_forecasts)

    # Custom complex model
    custom_model = Arima(
        ap,
        order=(2, 1, 3),
        seasonal={"order": (3, 2, 5), "period": 12},
        method="CSS-ML",
    )
    custom_model_forecasts = forecast_arima(custom_model, h=12)["mean"]
    assert_forward(custom_model, custom_model_forecasts)


# forecast_arima(forward_arima(custom_model, y=np.arange(1, 101)), h=12)["mean"]


# model = AutoARIMA()
# model = model.fit(ap)
# model.predict(h=7)
# model.predict(h=7, level=80)
# model.predict(h=7, level=(80, 90))
# model.predict_in_sample()
# model.predict_in_sample(level=50)
# model.predict_in_sample(level=(80, 50, 90)).tail(10).plot()
# model.model_.summary()
# model.summary()
# model_x = AutoARIMA(approximation=False)
# model_x = model_x.fit(ap, np.hstack([np.sqrt(drift), np.log(drift)]))
# model_x.predict(
#     h=12, X=np.hstack([np.sqrt(newdrift), np.log(newdrift)]), level=(80, 90)
# )
# model_x.predict_in_sample()
# model_x.predict_in_sample(level=(80, 90))
# model_x.summary()


def test_AutoARIMA_edge_cases(almost_constant_x):
    """Test AutoARIMA with various edge cases and data types."""
    # Test with constant array
    AutoARIMA().fit(np.array([1] * 36)).predict(20, level=80)

    # Test with single value
    number = 4.0
    preds = AutoARIMA().fit(np.array([number])).predict(3)
    np.testing.assert_array_equal(preds["mean"], number)

    # Test with almost constant data
    np.testing.assert_allclose(
        AutoARIMA().fit(almost_constant_x).predict(1).loc[0, "mean"],
        almost_constant_x.mean(),
    )


def test_auto_arima_with_trend():
    """Test auto_arima_f with trend (exogenous variables)."""
    trend = np.arange(ap.size, dtype=np.float64)
    model = auto_arima_f(
        ap,
        stepwise=False,
        xreg=trend.reshape(-1, 1),
    )
    np.testing.assert_equal(model["xreg"][:, 0], trend)


# ---------------------------------------------------------------------------
# Innovation-distribution tests
# ---------------------------------------------------------------------------

def _simulate_ar1(phi, innovations):
    """AR(1): y[t] = phi*y[t-1] + e[t], using scipy lfilter for speed."""
    return lfilter([1.0], [1.0, -phi], innovations)


# (distribution, rvs_kwargs, param_key, true_val, atol, n)
# Notes on tolerance choices:
#   t(df=5):  SE(nu_hat) ≈ 0.5 at n=1000; atol=1.0 covers ~2 SDs reliably.
#   ged(β=1): GED shape is well-identified; atol=0.25 at n=500 is stable.
#   ged(β=2): same, atol=0.30.
#   skew-normal is excluded here — its likelihood is multimodal (Azzalini 2013):
#   starting from alpha=0 the optimizer can converge to the wrong local minimum
#   regardless of n. It is tested separately via AIC improvement (see below).
_RECOVERY_CASES = [
    ("t",   {"df": 5},     "nu",        5.0, 1.0,  1000),
    ("ged", {"beta": 1.0}, "beta_dist", 1.0, 0.25,  500),
    ("ged", {"beta": 2.0}, "beta_dist", 2.0, 0.30,  500),
]


@pytest.mark.parametrize("distribution,rvs_kwargs,param_key,true_val,atol,n", _RECOVERY_CASES)
def test_distribution_parameter_recovery(distribution, rvs_kwargs, param_key, true_val, atol, n):
    """MLE should recover the true innovation-distribution parameter within tolerance."""
    rng = np.random.default_rng(0)
    dist_map = {"t": t_dist, "ged": gennorm}
    innovations = dist_map[distribution].rvs(**rvs_kwargs, size=n, random_state=rng)
    y = _simulate_ar1(0.6, innovations)

    fit = Arima(y, order=(1, 0, 0), distribution=distribution, method="ML")

    assert fit["distribution"] == distribution
    assert param_key in fit, f"key '{param_key}' missing from model dict"
    assert abs(fit[param_key] - true_val) < atol, (
        f"{distribution} {param_key}: got {fit[param_key]:.4f}, "
        f"expected {true_val} ± {atol}"
    )


def test_skewnorm_aic_improvement():
    """Skew-normal should reduce AIC vs normal when data has real skewness.

    The skew-normal likelihood is multimodal so magnitude recovery is unreliable,
    but the optimizer must at least find a mode that beats the normal model.
    """
    rng = np.random.default_rng(0)
    # Use large alpha so the skewness signal is strong and seed=0 lands on the right mode
    innovations = skewnorm.rvs(a=5, size=1000, random_state=rng)
    y = _simulate_ar1(0.6, innovations)

    fit_sn = Arima(y, order=(1, 0, 0), distribution="skew-normal", method="ML")
    fit_n  = Arima(y, order=(1, 0, 0), method="ML")

    assert fit_sn["aic"] < fit_n["aic"], (
        f"skew-normal AIC {fit_sn['aic']:.2f} should beat normal AIC {fit_n['aic']:.2f}"
    )
    assert fit_sn["alpha_dist"] > 0, "alpha_dist should be positive for right-skewed data"


def test_distribution_t_on_gaussian_gives_large_nu():
    """Fitting t-distribution on Gaussian data should yield nu >> 10."""
    rng = np.random.default_rng(1)
    y = _simulate_ar1(0.5, rng.standard_normal(500))
    fit = Arima(y, order=(1, 0, 0), distribution="t", method="ML")
    assert fit["nu"] > 10, f"Expected large nu on Gaussian data, got {fit['nu']:.2f}"


def test_distribution_ged_on_gaussian_gives_beta_near_2():
    """Fitting GED on Gaussian data should yield beta ≈ 2."""
    rng = np.random.default_rng(2)
    y = _simulate_ar1(0.5, rng.standard_normal(500))
    fit = Arima(y, order=(1, 0, 0), distribution="ged", method="ML")
    assert abs(fit["beta_dist"] - 2.0) < 0.3, (
        f"Expected beta_dist≈2 on Gaussian data, got {fit['beta_dist']:.4f}"
    )


@pytest.mark.parametrize("distribution,extra_key", [
    ("normal",      None),
    ("laplace",     None),
    ("t",           "nu"),
    ("skew-normal", "alpha_dist"),
    ("ged",         "beta_dist"),
])
def test_distribution_model_dict_keys(distribution, extra_key):
    """Each distribution stores exactly the right extra key (and not others)."""
    rng = np.random.default_rng(3)
    y = _simulate_ar1(0.4, rng.standard_normal(200))
    fit = Arima(y, order=(1, 0, 0), distribution=distribution, method="ML")

    assert fit["distribution"] == distribution
    assert "sigma2" in fit
    assert "nu" not in fit["coef"], "'nu' must not be in coef (breaks predict_arima)"
    assert "alpha_dist" not in fit["coef"]
    assert "beta_dist" not in fit["coef"]

    for key in ("nu", "alpha_dist", "beta_dist"):
        if key == extra_key:
            assert key in fit, f"Expected '{key}' in model for distribution='{distribution}'"
        else:
            assert key not in fit, (
                f"Unexpected key '{key}' in model for distribution='{distribution}'"
            )


@pytest.mark.parametrize("distribution", ["t", "skew-normal", "ged", "laplace"])
def test_distribution_css_raises(distribution):
    """Non-normal distributions must reject method='CSS'."""
    rng = np.random.default_rng(4)
    y = rng.standard_normal(100)
    with pytest.raises(ValueError, match="CSS"):
        arima(y, order=(1, 0, 0), distribution=distribution, method="CSS")


def test_distribution_invalid_name_raises():
    """Unknown distribution name raises ValueError."""
    rng = np.random.default_rng(5)
    y = rng.standard_normal(100)
    with pytest.raises(ValueError):
        arima(y, order=(1, 0, 0), distribution="cauchy")


@pytest.mark.parametrize("distribution,param_key", [
    ("t",           "nu"),
    ("skew-normal", "alpha_dist"),
    ("ged",         "beta_dist"),
])
def test_distribution_no_arma_params(distribution, param_key):
    """ARIMA(0,1,0) has no free ARMA params; sigma and shape must still be estimated."""
    rng = np.random.default_rng(6)
    y = np.cumsum(rng.standard_normal(200))
    fit = arima(y, order=(0, 1, 0), distribution=distribution, method="ML")

    assert fit["distribution"] == distribution
    assert param_key in fit
    assert np.isfinite(fit[param_key])
    assert fit["sigma2"] > 0


@pytest.mark.parametrize("distribution,param_key", [
    ("t",           "nu"),
    ("skew-normal", "alpha_dist"),
    ("ged",         "beta_dist"),
])
def test_distribution_autoarima_threads(distribution, param_key):
    """auto_arima_f must carry distribution through the re-fitting step."""
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.standard_normal(100))
    fit = auto_arima_f(y, distribution=distribution, method="ML")

    assert fit["distribution"] == distribution
    assert param_key in fit
    assert np.isfinite(fit[param_key])


@pytest.mark.parametrize("distribution,param_key", [
    ("t",           "nu"),
    ("skew-normal", "alpha_dist"),
    ("ged",         "beta_dist"),
])
def test_distribution_forecast_intervals(distribution, param_key):
    """forecast_arima must produce finite, ordered intervals for every distribution."""
    rng = np.random.default_rng(8)
    y = _simulate_ar1(0.5, rng.standard_normal(200))
    fit = Arima(y, order=(1, 0, 0), distribution=distribution, method="ML")
    fc = forecast_arima(fit, h=10, level=[80, 95])

    lower_80 = fc["lower"]["80%"].values
    upper_80 = fc["upper"]["80%"].values
    lower_95 = fc["lower"]["95%"].values
    upper_95 = fc["upper"]["95%"].values

    assert np.all(np.isfinite(lower_95))
    assert np.all(np.isfinite(upper_95))
    assert np.all(lower_80 > lower_95), "95% lower bound must be below 80%"
    assert np.all(upper_80 < upper_95), "95% upper bound must be above 80%"
    assert np.all(lower_95 < upper_95), "lower must be below upper"


def test_issue_649(capsys):
    from statsforecast.models import AutoARIMA

    y = np.array(42 * [100] + [119, 525])
    AutoARIMA(season_length=12, trace=True).fit(y)
    captured = capsys.readouterr()
    expected_output = """ARIMA(2,0,2)(1,0,1)[12] with non-zero mean : inf
ARIMA(0,0,0)            with non-zero mean : 494.2237
ARIMA(1,0,0)(1,0,0)[12] with non-zero mean : inf
ARIMA(0,0,1)(0,0,1)[12] with non-zero mean : inf
ARIMA(0,0,0)            with zero mean     : 553.2571
ARIMA(0,0,0)(1,0,0)[12] with non-zero mean : 496.5234
ARIMA(0,0,0)(0,0,1)[12] with non-zero mean : 496.5226
ARIMA(0,0,0)(1,0,1)[12] with non-zero mean : inf
ARIMA(1,0,0)            with non-zero mean : inf
ARIMA(0,0,1)            with non-zero mean : 494.3700
ARIMA(1,0,1)            with non-zero mean : inf
"""
    assert captured.out == expected_output


def test_issue_1167():
    data_raw = "2.0785799311843243\n5.465227016278618\n7.21496098169046\n8.804593399364494\n9.824441222195027\n10.351057764215522\n9.796232926707928\n8.568689743748763\n7.207535518526844\n5.771192550213702\n2.8193330414176283\n0.8337842503680846\n-1.4083119185089308\n-2.7890031410796405\n-2.7609698747667757\n-2.04328567594115\n-1.084043524900012\n0.6897641380813886\n1.866072143822579\n2.9672606190512294\n3.083214884954371\n3.710445233127744\n4.505071409634818\n3.7893779818993805\n3.7661984172329417\n4.327600750596117\n5.098806942229123\n3.4282521312862237\n1.7220490254057736\n-0.7464939997741948\n-3.0046115778807905\n-4.783308271042684\n-5.7509268496286605\n-5.582407234746615\n-5.007057925921204\n-3.1564071666509728\n-2.0903140862882887\n-1.0395687744320286\n0.9229359639241858\n2.469991022766959\n3.6285146679191533\n3.269595006438753\n2.724764818201851\n2.0181018587576034\n0.6333565809568373\n-0.8337261443085004\n-2.673953635394059\n-6.207505657045569\n-10.012875163075725\n-11.518639072825728\n-10.292984866163781\n-7.604461610724283\n-5.304059495412831\n-3.1831934099519774\n-0.03248367336499591\n2.3757843707908757\n4.348780661662822\n6.493148053420233\n8.399490589257981\n10.054576479207181\n10.820739665923098\n10.506826989577153\n9.625286143925775\n5.518822509018423\n2.3301643233482734\n0.3271296328958202\n-2.2150280780578475\n-3.770341434882302\n-3.998690794601169\n-3.7782753056987906\n-2.038678451925746\n-0.5829458954918092\n1.8051249831366996\n4.1513689484177\n5.774641510900511\n4.855783962680863\n3.0557908047452074\n0.05448973305695626\n-3.3941719767828\n-5.400856019606619\n-6.6076992465861935\n-5.817856711374767\n-4.395993449799106\n-4.348501028993221\n-3.246693391963909\n-1.913054635642962\n-1.7448925200560188\n-0.6779805213800589\n1.1385459630458623\n3.330827167096523\n5.266065409093031\n4.470815676909511\n4.045505099229681\n3.615671629129498\n2.362747269599752\n-1.2594592021425566\n-3.622631693392976\n-6.043024177586433\n-8.958588711845746\n-12.036009458202873\n-12.698517267109427\n-12.676333005814717\n-12.5841893417497\n-11.248449792234915\n-7.795023837906228\n-3.510496638346101\n0.280009638397539\n4.572900779528343\n6.991416847541994\n7.77895770987047\n8.570770170511935\n9.067940226095159\n9.604868167138257\n9.923146851720984\n6.385793282268368\n3.616796545102801\n1.9123482043754705\n0.752370132452693\n-0.6538290745466866\n-1.1296707915386373\n-2.2466460593576136\n-3.1500342902450056\n-4.385746095280917\n-4.119263208000346\n-4.070101080194862\n-3.1322695965473546\n-1.7067903035327547\n0.08966128874792434\n2.721620572982131\n5.431564556909196\n6.599144895769013\n7.1490778339267\n6.43120580376761\n5.947200121441169\n4.6125847526500925\n3.3711148644785562\n2.6169130252498753\n2.418169806625445\n3.048799808641145\n4.283325309657206\n5.707114495204862\n7.160680931931025\n8.11195238273421\n8.094701816683198\n8.100546115024006\n6.376379440532032\n4.801440715091021\n3.0309454598643635\n2.1331748186354873\n1.4486954746260376\n1.2149517306350948\n-0.01664094886867762\n-0.10024812437292707\n-0.23211717792862013\n-1.107086136408511\n-0.7258253777743642\n-0.21383719420706637\n-0.616826410239954\n-3.9570531100890065\n-6.987000503992364\n-11.460448663932532\n-14.83181086328094\n-15.495553671718845\n-16.016888181923232\n-14.223762053908015\n-11.170994254009624\n-7.689816857012993\n-4.19736194936186\n-1.3751814565748468\n2.014837904694427\n5.700916044619497\n8.444955799304827\n10.993878605320166\n12.755869155206412\n13.14318488252167\n11.893613111790303\n8.475205686925568\n5.264640857350619\n-1.0143479013261718\n-6.514357092144627\n-9.816130860423986\n-12.03204383197698\n-12.387226352518212\n-11.613823607716201\n-9.58670470783021\n-6.883260229789718\n-3.2318472401803193\n0.5489501120258291\n3.830989213133614\n5.7125687714445\n7.265515087233187\n6.676344996805199\n4.557880039104962\n1.9993154782190634\n-0.06794005843760609\n-0.912250659968739\n-0.6274360316490972\n0.7433672198900281\n2.9366328651016227\n5.5225601528428525\n7.810327096755828\n9.845653148121746\n8.905426139709107\n7.863658824031061\n7.122585396126469\n6.571348986625629\n5.914732933822529\n4.983306806743025\n3.6011201540227904\n2.5515088991210586\n1.8290051362755086\n1.6445101907730035\n1.0573023216189639\n1.0088204448215916\n0.9113445090082065\n1.7310155272886227\n2.2603766627852444\n3.1037746495749365\n3.239859800185876\n3.5945238215099358\n3.2660473333421822\n3.481089804519456\n1.5323201786673084\n-1.1374542408905586\n-3.4563666802165915\n-5.973126850342381\n-6.4965425745564085\n-5.733326274524046\n-4.039837209105459\n-1.3227767767308465\n1.6898518131248976\n5.092252752251268\n8.232483225278406\n10.479251844824924\n12.155498860615284\n12.826132956446855\n10.901616134991485\n8.296131117981913\n5.637602730865883\n3.126481550746079\n0.9102430295859505\n-1.0288148912654553\n-3.963401861631775\n-8.853197141002902\n-12.273035948155464\n-14.177124927467338\n-13.283887399723248\n-10.772799907276296\n-6.6407714651605225\n-1.8232973988496908\n3.5730414227342755\n8.326809167893096\n12.109566805059362\n13.596893964952107\n14.023901653876841\n13.354550366303558\n11.293663815463397\n8.942791814656863\n6.528322497048997\n3.374304856188852\n0.10696073525162242\n-3.746399603719927\n-9.712615800736842\n-14.091823255676676\n-15.434393354627828\n-15.220680630062652\n-13.060790410128002\n-9.037120458820347\n-3.832793773188044\n0.7133889486820322\n5.371218180198104\n8.30806889510927\n10.310502826439567\n7.698422567629633\n3.710022767396219\n-0.4098087605773375\n-3.9865389803233295\n-5.672839037329086\n-5.339107788955871\n-7.027551048172203\n-6.782640140278678\n-6.026873781879357\n-4.389385608212468\n-3.4652130292017103\n-2.426345598197263\n-2.508373344310072\n-2.2140176801834413\n-1.4310315503964173\n-1.931867027639038\n-1.7812536858438315\n-0.6849806827828564\n0.8335113017876056\n-4.938819015594646\n-9.042262960494813\n-11.098139941137466\n-11.201224928312083\n-9.104701666545084\n-7.210576250646717\n-4.320897831810753\n-1.6288517972893921\n0.6863865053526148\n3.206523727200905\n5.9390160236768175\n7.716410548712698\n8.281993346938975\n8.39532767571353\n5.856653366678383\n3.062140504200121\n0.14224419696224677\n-1.540072596069932\n-2.5029483042334713\n-3.55110077798991\n-3.5174264087626486\n-2.1303755393044006\n-0.6176276785193795\n1.500533550162132\n3.635850663709653\n5.312032724975669\n6.898630661878205\n5.785704921672531\n2.457694988556215\n-0.4832173096965169\n-2.060225410425332\n-4.332425997660267\n-7.022749632199073\n-7.657038726157958\n-7.5225133616765545\n-6.8580395708460005\n-6.014663337513218\n-4.914273895188575\n-4.540921145567153\n-5.972630605889824\n-5.864433715168832\n-5.626266341464111\n-4.941917252282045\n-3.362244770617029\n-1.767634239241583\n0.7608258610222143\n2.61370560107527\n2.7283337793216687\n1.7733327245769301\n1.4409474399810296\n1.4942998584556626\n2.289343461082176\n2.1654730681026892\n2.81728847557818\n3.7058685146659927\n4.0010509308991455\n3.57197471825548\n3.1019254709911226\n-0.33975191125275206\n-2.2225831161956613\n-4.099354188622683\n-6.288077457981693\n-7.001320107075809\n-6.662689858290192\n-5.91735824788941\n-5.349877604126496\n-5.009032559674794\n-3.077714548407987\n-0.47858027031717343\n1.9435419169508634\n4.681395002848878\n5.768607256802374\n5.765320349001861\n5.6490642374081785\n5.219056238564082\n4.230237549862461\n2.973322154881328\n1.514326407079789\n1.0823228441425536\n-1.2864333400720014\n-3.940169766533064\n-5.772092451863783\n-6.088735918518408\n-4.887249616450954\n-2.9530012997847574\n-0.5000637444952246\n-2.092269174054738\n-2.266955988764141\n-3.269455286842829\n-2.8895535648579838\n-1.1240168289031096\n1.1595861138171395\n2.710370916738409\n4.842785805899464\n5.251202975293031\n5.827687447166869\n6.538838828350883\n5.321246431044948\n4.10592643180231\n2.3345355617534134\n0.5023606565919456\n-1.1896775245784352\n-3.365802053990073\n-3.796616029961279\n-2.868923710151585\n-1.6931158449490513\n-0.4980507345941758\n0.438558261172421\n1.9246106267486693\n3.070778609675494\n3.6319023464155027\n4.651944163329084\n3.238879257199755\n0.028118140881418174\n-1.9876652296700352\n-3.259734399633788\n-3.138928637174942\n-1.6030414786769882\n1.0228883020384234\n4.18443176896325\n7.287560339831029\n8.522435081677303\n9.099186889712556\n9.275612007674306\n6.8009563593587306\n3.1965344504318622\n0.61853528702919\n-0.7722146724725101\n-1.4752980958011106\n-3.2770461777563624\n-3.749982289266252\n-3.971416646906544\n-3.1964541124668195\n-3.2279661166138345\n-1.9057271780939171\n0.6180326777089418\n3.8146674086396417\n7.158362143932763\n10.151496557708548\n12.2802034063049\n13.723051063967555\n13.090959767239633\n12.01608302505717\n10.015772713255572\n6.887594831126272\n3.4979136502442048\n-0.03038638726334142\n-3.151944433428077\n-5.972093156270136\n-9.265231012582618\n-12.053292633179948\n-12.496502311096112\n-11.237390900269657\n-8.147846883556676\n-3.501291745883548\n0.9537006745865388\n5.451685157188831\n9.41712416842238\n12.417724891467973\n14.418524656056329\n15.174199813456255\n13.681886286891444\n10.531395172886779\n7.3896330515286355\n4.3899839523280955\n0.8635192790944048\n-2.843094431318489\n-5.134147958718284\n-5.685387404811742\n-11.423349772675756\n-14.71531727709702\n-15.089616796762458\n-13.408917378993815\n-11.717465578165514\n-8.210184968769214\n-3.2323534814582624\n2.4836035376218977\n7.65265609948683\n12.119119790884906\n15.210929953644222\n15.838649229374179\n15.710742258933346\n14.781121578657821\n12.676173448206836\n9.637778578979809\n5.883656223394997\n2.3287063552292206\n-1.3047426247296123\n-3.8290242242062464\n-4.590126986631178\n-5.1536529732012815\n-5.862150281631212\n-8.409615743252713\n-9.090310878063683\n-9.73802525972031\n-8.596619609595242\n-5.6233047486512096\n-1.416342410905954\n3.5045985051431536\n8.006839402672082\n11.968850221835831\n14.706098983530916\n15.97846511708697\n16.093291341094787\n14.387538749876956\n11.807479756262914\n7.898168723144849\n4.671627608431881\n1.908204656408585\n0.3576672734308435\n-1.9428586942975403\n-2.6209274499923394\n-2.429149990619413\n-1.7103788135477527\n-1.0243652479529204\n0.07730058811056173\n1.4348758535768855\n3.2642694386798587\n4.350464746037839\n3.746254577634028\n3.3572921052227453\n0.5087633712307245\n-2.6596635509587774\n-5.095609013460472\n-6.808964899523036\n-7.245083882974361\n-6.059397541505682\n-3.5223005709460766\n-1.7786688157243742\n0.4130914407712062\n2.895941382760047\n5.376670485466505\n7.737469566018329\n8.887950378995429\n8.910593748086903\n8.322125219111388\n7.401182572383676\n6.670466764768537\n5.440924710157273\n4.440666692558761\n3.617300403395352\n3.055843910371934\n3.1121171153309173\n2.296341599061975\n1.326899076901108\n-0.3240171722090568\n-1.5909415403054976\n-3.9078203815358945\n-4.512706309797236\n-4.523154873278139\n-3.178657370526085\n-1.4348253621118459\n1.3240562141724463\n4.6786652212346915\n5.387909355193752\n5.014081749943465\n4.752003521290893\n2.4188471458349023\n0.7186130817640632\n-0.4735895947963801\n-1.626610184026758\n-3.077853008825884\n-3.3063783351633957\n-2.9461207493503543\n-2.610039644538179\n-0.9958280365804035\n1.381356090012254\n3.8249301829598394\n4.2744659073437745\n3.4326955578277554\n1.8357294563840048\n0.2659057779422971\n-1.2969250126150247\n-2.85184756664212\n-3.617976932442553\n-4.901864087866953\n-6.456716139920698\n-6.351818810922354\n-4.4463524248502715\n-1.4170609817934388\n1.826429236642169\n5.189308325546052\n8.590335667928187\n11.462035309953567\n12.41331657647647\n12.840184700647907\n10.509283674981372\n7.951026027218299\n5.051187893608271\n1.8325028660784708\n-0.566429867867396\n-1.9102640538134676\n-1.7697134678368747\n-2.4170581332531844\n-3.017215352162019\n-3.4380150171484036\n-2.5282595042618903\n-0.677872565228393\n2.043396596260943\n3.360917508337679\n5.086193062713088\n6.317044084582541\n7.675435425800673\n7.415955272371262\n6.926945403666469\n0.664202812242527\n-5.315223468729174\n-11.606168979175374\n-15.059114375910074\n-18.22424569255403\n-17.923928966682617\n-15.197961826336021\n-10.665601049169823\n-7.171919588774646\n-3.1540230368957634\n-0.5034349247656338\n1.4625601629823306\n3.915048854705838\n5.499120583526475\n6.700029720802637\n7.428872383627377\n8.344528896067594\n7.922663681806913\n7.698243237953417\n7.71426798608444\n7.5035312540414765\n6.93674729627018\n4.019922564578016\n1.7948426014017043\n0.4785135075112519\n-0.5007483435546924\n-1.0264142505935352\n-2.0476354719272916\n-4.517283647228088\n-6.753952200209429\n-8.191047820556603\n-8.680933761294693\n-12.472380155241042\n-14.51997665099847\n-14.12619660140222\n-15.602297448331777\n-15.562358556431942\n-14.314091620462332\n-13.315258037878998\n-9.846507365748185\n-4.736040193650154\n0.030489415047664448\n4.835084879032185\n8.574459460162478\n11.87610230035406\n11.663784066047807\n10.841755482301338\n7.635491850554247\n5.098966399504012\n2.6447552635846012\n1.1090361598175962\n0.5062978168231891\n0.264859314349016\n-0.6994832717671079\n-1.573441658193605\n-1.4885520183558576\n-0.21310642657511214\n1.429117719975332\n3.1060713492757355\n3.0451199320424664\n3.093276668383243\n1.5542031209197398\n0.7702416452783436\n0.989900871580371\n0.4620841449493489\n0.3755694141180841\n1.265625351188814\n2.4452846708008797\n3.3848778469544625\n3.9602845853671003\n4.731450008416438\n5.861712089135541\n6.376071964056612\n5.416465135330918\n5.021701611130584\n4.970430529406379\n5.448747010015948\n4.117556043549144\n2.23315357899237\n0.9972193996014059\n0.466046544707418\n-1.2617934029608833\n-2.995716671259939\n-3.889066783464387\n-4.69179154857396\n-4.454418931979522\n-3.498676427998512\n-3.5514142682313543\n-3.777157954821334\n-6.263205212614118\n-8.188493031565457\n-7.793384585591307\n-5.9449234222268\n-3.3511741248088502\n-0.9150991106916995\n1.849602496297826\n4.561842901941533\n6.738317732236454\n9.046132749828473\n10.638111341332309\n11.20198852920032\n10.95781030516013\n10.62622890391471\n8.763982452430374\n6.562635160297078\n4.397946160241011\n0.6295498463644673\n-3.0047754183442406\n-4.906601571041134\n-6.293519701305762\n-6.4629296760817185\n-5.5953465655063255\n-3.1249472670737246\n-0.4861798447759913\n2.7601080223756966\n5.189392544225312\n7.776515652428096\n8.372403575458257\n9.028635457748477\n9.561623235335517\n8.861637661552159\n8.290272301741915\n6.989286414910603\n6.017885512635322\n3.068405416561986\n1.1080347440353284\n0.16163820523981476\n-0.7970294743973306\n-2.802415170742049\n-3.1560374707083385\n-2.7693999401437153\n-1.4603852681498133\n0.5074954895301476\n1.3301327761663209\n2.768888394888855\n4.794181365171442\n6.461700144384579\n8.04060438812964\n7.160112852464152\n4.9001824354911765\n3.4970324876262335\n2.544140083943305\n2.504038281323476\n2.8895172392842623\n2.0284892624423887\n0.7003998563350726\n-0.6658015445850567\n-1.6254494211033108\n-2.563683485652934\n-3.4872149657973543\n-6.681966643561958\n-8.908427721766692\n-10.602389081771845\n-10.113695407627754\n-9.3869959798473\n-7.1968434879542755\n-3.756228556420999\n0.02021608863767599\n3.4051596342676573\n4.191061094362258\n4.65134756593451\n4.253441586561111\n3.078279508522791\n0.39209203445819707\n-1.4545032293549465\n-3.0244357394701504\n-3.7157773384061974\n-2.883201107973359\n-1.5272487260322158\n-0.22220899982456377\n0.2836882217928224\n0.5261473084287789\n1.6852268449266057\n2.5963983958621446\n3.569639869773341\n4.557158191370504\n4.591142755862293\n4.395747870005259\n3.8928691305203134\n4.045863374481471\n3.5076874451621953\n2.8213965847764104\n2.4333215729681292\n2.388818947553265\n0.9732480591791082\n-0.7657502237740108\n-1.1730149200809654\n-0.25581710594450535\n1.202419820595352\n1.9146750280414517\n0.5485931475978845\n0.4327989948615236\n-0.04291916218396824\n0.41031339504578584\n-0.3542357680776482\n-0.29224414751255834\n0.29193876704910793\n0.6092614819390443\n-0.055081398453353025\n0.40555509278861746\n0.3954511528902521\n1.2456521985048723\n1.4962629247758312\n1.1437715553223158\n1.7131254237745366\n2.927613270613307\n4.42678766164474\n6.32009979105213\n6.89182412209608\n6.551696814973273\n6.298071839045957\n5.4132083905669335\n4.820775660760005\n4.49736621961767\n2.113743372391019\n-0.046852740472004495\n-0.856143679558663\n-1.278652600035568\n-3.1063479888295733\n-3.485437158573464\n-2.780944087835862\n-1.2833294562966786\n1.1904960640547286\n3.0176075036554506\n4.623160004739477\n4.41548720304092\n2.6621024757193505\n0.41056381930068864\n-1.9417249278713704\n-2.8213702321352114\n-3.00958457604826\n-2.3973662529076587\n-5.137884385694223\n-8.069908231370901\n-11.829239642471105\n-15.824296622970003\n-18.814490065384923\n-21.641501759023818\n-22.677119207483987\n-19.968366148830953\n-17.693885629687326\n-12.904522012103147\n-6.90474465077904\n-0.2638216558044977\n5.990839523559183\n10.200515688376989\n13.294392845835256\n14.383112827636328\n14.595941839962702\n13.94637334227525\n10.64850877322582\n6.6645230049719855\n1.8843190107847132\n-1.4204136995576089\n-4.478309013610413\n-6.2000032041744415\n-5.90292502520595\n-4.4666455850879245\n-1.7431062569020908\n0.8121985901536068\n2.4972947413366766\n4.192807132885439\n6.0031411720877434\n6.631097551851498\n7.531379263697958\n7.556899450329865\n7.4327617035745215\n7.0840518428921335\n6.215782804543659\n3.760904028024908\n2.053286023847315\n0.6470263654750378\n0.2859915554937462\n0.7110488161943045\n0.8239542123299344\n1.191125902473074\n2.094834836511154\n2.791261536955586\n3.23583007773402\n1.565580794336228\n-0.8546710821298176\n-1.8546777501303473\n-2.940572195532859\n-4.75415067286837\n-6.297077770909032\n-9.082367791531842\n-9.66551190405699\n-9.382954575755903\n-7.257931523586496\n-3.8308395770536436\n0.6890359350215935\n4.624491662148623\n5.110539508227593\n4.344374176447976\n4.231425500047646\n4.1291040162603725\n4.388122881833323\n2.8990408481811185\n0.7450778394945003\n-5.428118381846088\n-9.079025978135004\n-11.538599607686146\n-12.107552625033406\n-11.600100051320663\n-9.461194893217327\n-5.583156041883045\n-0.6963948366101651\n4.239128557197372\n7.093474709651765\n8.314365951586161\n7.828982422150158\n6.450472988779054\n4.772912505099764\n3.7050169554266073\n3.2375556814578577\n2.9086031165558115\n2.151453308994088\n1.7240999329160172\n0.6389146938300383\n-0.2958307983212586\n-0.5585385900577662\n0.14075097015043825\n-0.10934830996988243\n0.43635487853379906\n1.170792206229395\n1.7888607476463814\n2.8909988067064316\n4.425135478207728\n5.928669191073469\n6.433574248909926\n5.522657348753311\n4.093725450127535\n2.1933395972957\n0.6851965072332541\n0.026676096032100416\n-0.5328899134342205\n0.1374104873212434\n-0.11797860594486642\n-1.3766667404463582\n-2.1229647328188217\n-1.9336125341933332\n-1.4257627403058644\n-0.236714424340845\n1.6968872221584625\n3.6815238680034588\n6.009377887588035\n6.969205662442722\n7.215842112312965\n6.887154126972094\n6.554963994565091\n5.635831542540089\n4.397468835360627\n2.7607767074064054\n1.6668531220361857\n1.0344811443814081\n1.134145018801395\n0.902235594089319\n1.3659414695679901\n1.2886702685783678\n1.906732926307505\n1.5222806650311818\n1.5188491954657821\n2.1027924827232733\n2.288844317175819\n3.319218963820227\n4.722204137101622\n4.572313026342773\n3.5769638778493054\n-0.767732202061012\n-5.129222407673913\n-7.99243618224118\n-9.449046805501323\n-9.636039545770583\n-7.673026148230354\n-4.472081234106204\n-1.1906431925033754\n0.8758252388327716\n3.507496552136328\n5.442927100269059\n6.381314989974585\n5.785975351143332\n5.4129116664443595\n3.774552531515806\n0.22325932068045518\n-2.1351764460782654\n-3.21470474665733\n-7.983408362827211\n-12.675933775708188\n-14.530809100506044\n-14.714319963988933\n-13.775258363496988\n-12.09643308331555\n-8.837879091201346\n-4.431687276263828\n1.075556696712276\n7.014002716998922\n12.306718531211382\n13.170932974667002\n12.17935308431019\n8.759617297687008\n5.024254037636993\n-1.0361761280150823"
    d = [float(i) for i in data_raw.splitlines()]
    d = np.array(d)
    mdl = ARIMA(order=(2, 0, 0))
    mdl.fit(d)
    coef = mdl.model_["coef"]
    assert math.isclose(coef["ar1"], 1.8, abs_tol=0.1)
    assert math.isclose(coef["ar2"], -0.9, abs_tol=0.1)
    assert math.isclose(mdl.model_["sigma2"], 0.9, abs_tol=0.1)
