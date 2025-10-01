import math
import warnings

import numpy as np
import pytest

# from fastcore.test import test_close, test_eq
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
