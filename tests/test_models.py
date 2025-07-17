from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# from fastcore.test import test_close, test_eq, test_fail
from statsforecast.garch import generate_garch_data
from statsforecast.models import (
    _TS,
    ADIDA,
    ARCH,
    ARIMA,
    GARCH,
    IMAPA,
    MSTL,
    TSB,
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoMFLES,
    AutoRegressive,
    AutoTBATS,
    AutoTheta,
    ConformalIntervals,
    ConstantModel,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    DynamicOptimizedTheta,
    DynamicTheta,
    HistoricAverage,
    Holt,
    HoltWinters,
    Naive,
    NaNModel,
    OptimizedTheta,
    RandomWalkWithDrift,
    SeasonalExponentialSmoothing,
    SeasonalExponentialSmoothingOptimized,
    SeasonalNaive,
    SeasonalWindowAverage,
    SimpleExponentialSmoothing,
    SimpleExponentialSmoothingOptimized,
    SklearnModel,
    WindowAverage,
)
from statsforecast.utils import AirPassengers as ap


def _plot_insample_pi(fcst):
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))

    sdate = date(1949, 1, 1)  # start date
    edate = date(1961, 1, 1)  # end date
    dates = pd.date_range(sdate, edate - timedelta(days=1), freq="m")

    df = pd.DataFrame(
        {
            "dates": dates,
            "actual": ap,
            "fitted": fcst["fitted"],
            "fitted_lo_80": fcst["fitted-lo-80"],
            "fitted_lo_95": fcst["fitted-lo-95"],
            "fitted_hi_80": fcst["fitted-hi-80"],
            "fitted_hi_95": fcst["fitted-hi-95"],
        }
    )
    plt.plot(df.dates, df.actual, color="firebrick", label="Actual value", linewidth=3)
    plt.plot(df.dates, df.fitted, color="navy", label="Fitted values", linewidth=3)
    plt.plot(
        df.dates, df.fitted_lo_80, color="darkorange", label="fitted-lo-80", linewidth=3
    )
    plt.plot(
        df.dates,
        df.fitted_lo_95,
        color="deepskyblue",
        label="fitted-lo-95",
        linewidth=3,
    )
    plt.plot(
        df.dates, df.fitted_hi_80, color="darkorange", label="fitted-hi-80", linewidth=3
    )
    plt.plot(
        df.dates,
        df.fitted_hi_95,
        color="deepskyblue",
        label="fitted-hi-95",
        linewidth=3,
    )
    plt.fill_between(
        df.dates, df.fitted_lo_95, df.fitted_hi_95, color="deepskyblue", alpha=0.2
    )
    plt.fill_between(
        df.dates, df.fitted_lo_80, df.fitted_hi_80, color="darkorange", alpha=0.3
    )
    plt.legend()


def _plot_fcst(fcst):
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    plt.plot(np.arange(0, len(ap)), ap)
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst["mean"], label="mean")
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst["lo-95"], color="r", label="lo-95")
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst["hi-95"], color="r", label="hi-95")
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst["lo-80"], color="g", label="lo-80")
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst["hi-80"], color="g", label="hi-80")
    plt.legend()


# test conformity scores
class ZeroModel(_TS):
    def __init__(self, prediction_intervals: ConformalIntervals = None):
        self.prediction_intervals = prediction_intervals
        self.alias = "SumAhead"

    def forecast(self, y, h, X=None, X_future=None, fitted=False, level=None):  # noqa: ARG002
        res = {"mean": np.zeros(h)}
        if self.prediction_intervals is not None and level is not None:
            cs = self._conformity_scores(y, X)
            res = self._conformal_method(fcst=res, cs=cs, level=level)
        return res

    def fit(self, y, X):
        return self

    def predict(self, h, X=None, level=None):
        res = {"mean": np.zeros(h)}
        return res


def test_conformal_intervals():
    conf_intervals = ConformalIntervals(h=12, n_windows=10)
    expected_cs = np.full((conf_intervals.n_windows, conf_intervals.h), np.nan)
    cs_info = ap[-conf_intervals.h * conf_intervals.n_windows :]
    for i in range(conf_intervals.n_windows):
        expected_cs[i] = cs_info[i * conf_intervals.h : (i + 1) * conf_intervals.h]
    current_cs = ZeroModel(conf_intervals)._conformity_scores(ap)
    np.testing.assert_array_equal(expected_cs, current_cs)
    zero_model = ZeroModel(conf_intervals)
    fcst_conformal = zero_model.forecast(ap, h=12, level=[80, 90])
    assert list(fcst_conformal.keys()) == ["mean", "lo-90", "lo-80", "hi-80", "hi-90"]


def assert_class(
    cls_,
    x,
    h,
    skip_insample=False,
    level=None,
    test_forward=False,
    X=None,
    X_future=None,
):
    cls_ = cls_.fit(x, X=X)
    fcst_cls = cls_.predict(h=h, X=X_future)
    assert len(fcst_cls["mean"]) == h
    # test fit + predict equals forecast
    np.testing.assert_array_equal(
        cls_.forecast(y=x, h=h, X=X, X_future=X_future)["mean"], fcst_cls["mean"]
    )
    if not skip_insample:
        assert len(cls_.predict_in_sample()["fitted"]) == len(x)
        assert isinstance(cls_.predict_in_sample()["fitted"], np.ndarray)
        np.testing.assert_array_equal(
            cls_.forecast(y=x, h=h, X=X, X_future=X_future, fitted=True)["fitted"],
            cls_.predict_in_sample()["fitted"],
        )
        if test_forward:
            np.testing.assert_array_equal(
                cls_.forward(y=x, h=h, X=X, X_future=X_future, fitted=True)["fitted"],
                cls_.predict_in_sample()["fitted"],
            )

    if test_forward:
        try:
            pd.testing.assert_frame_equal(
                pd.DataFrame(cls_.predict(h=h, X=X_future)),
                pd.DataFrame(cls_.forward(y=x, X=X, X_future=X_future, h=h)),
            )
        except AssertionError:
            raise Exception("predict and forward methods are not equal")

    if level is not None:
        fcst_cls = pd.DataFrame(cls_.predict(h=h, X=X_future, level=level))
        fcst_forecast = pd.DataFrame(
            cls_.forecast(y=x, h=h, X=X, X_future=X_future, level=level)
        )
        try:
            pd.testing.assert_frame_equal(fcst_cls, fcst_forecast)
        except AssertionError:
            raise Exception("predict and forecast methods are not equal with levels")

        if test_forward:
            try:
                pd.testing.assert_frame_equal(
                    pd.DataFrame(cls_.predict(h=h, X=X_future, level=level)),
                    pd.DataFrame(
                        cls_.forward(y=x, h=h, X=X, X_future=X_future, level=level)
                    ),
                )
            except AssertionError:
                raise Exception("predict and forward methods are not equal with levels")

        if not skip_insample:
            fcst_cls = pd.DataFrame(cls_.predict_in_sample(level=level))
            fcst_forecast = cls_.forecast(
                y=x, h=h, X=X, X_future=X_future, level=level, fitted=True
            )
            fcst_forecast = pd.DataFrame(
                {key: val for key, val in fcst_forecast.items() if "fitted" in key}
            )
            try:
                pd.testing.assert_frame_equal(fcst_cls, fcst_forecast)
            except AssertionError:
                raise Exception(
                    "predict and forecast methods are not equal with "
                    "levels for fitted values "
                )
            if test_forward:
                fcst_forward = cls_.forecast(
                    y=x, h=h, X=X, X_future=X_future, level=level, fitted=True
                )
                fcst_forward = pd.DataFrame(
                    {key: val for key, val in fcst_forward.items() if "fitted" in key}
                )
                try:
                    pd.testing.assert_frame_equal(fcst_cls, fcst_forward)
                except AssertionError:
                    raise Exception(
                        "predict and forward methods are not equal with "
                        "levels for fitted values "
                    )


@pytest.mark.parametrize(
    "model_factory",
    [
        ADIDA,
        CrostonClassic,
        CrostonOptimized,
        CrostonSBA,
        IMAPA,
        lambda: TSB(alpha_d=0.9, alpha_p=0.1),
    ],
)
def test_fitted_sparse(model_factory):
    y1 = np.array([2, 5, 0, 1, 3, 0, 1, 1, 0], dtype=np.float64)
    y2 = np.array([0, 0, 1, 0, 0, 7, 1, 0, 1], dtype=np.float64)
    y3 = np.array([0, 0, 1, 0, 0, 7, 1, 0, 0], dtype=np.float64)
    y4 = np.zeros(9, dtype=np.float64)
    for y in [y1, y2, y3, y4]:
        expected_fitted = np.hstack(
            [
                model_factory().forecast(y=y[: i + 1], h=1)["mean"]
                for i in range(y.size - 1)
            ]
        )
        np.testing.assert_allclose(
            model_factory().forecast(y=y, h=1, fitted=True)["fitted"],
            np.append(np.nan, expected_fitted),
            atol=1e-6,
        )


class TestArima:
    fcst_arima = None

    def test_arima(self):
        arima = AutoARIMA(season_length=12)
        assert_class(arima, x=ap, h=12, level=[90, 80], test_forward=True)

        TestArima.fcst_arima = arima.forecast(ap, 13, None, None, (80, 95), True)

    def test_arima_conformal_prediction(self):
        # test conformal prediction
        arima_c = AutoARIMA(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(arima_c, x=ap, h=13, level=[90, 80], test_forward=True)
        fcst_arima_c = arima_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(
            fcst_arima_c["mean"],
            TestArima.fcst_arima["mean"],
        )

    def test_alias_arg(self):
        # test alias argument
        assert repr(AutoARIMA()) == "AutoARIMA"
        assert repr(AutoARIMA(alias="AutoARIMA_seasonality")) == "AutoARIMA_seasonality"


class TestETS:
    def test_ets(self):
        autoets = AutoETS(season_length=12)
        assert_class(autoets, x=ap, h=12, level=[90, 80], test_forward=True)
        # TestETS.fcst_ets = autoets.forecast(ap, 13, None, None, (80, 95), True)

    def test_alias_arg(self):
        # test alias argument
        assert repr(AutoETS()) == "AutoETS"
        assert repr(AutoETS(alias="AutoETS_custom")) == "AutoETS_custom"

    # def test_ets_conformal_prediction(self):
    #     autoets = AutoETS(season_length=12, model="AAA")
    #     test_class(autoets, x=ap, h=12, level=[90, 80])
    #     fcst_ets = autoets.forecast(ap, 13, None, None, (80, 95), True)
    #     _plot_insample_pi(fcst_ets)

    def test_ets_conformal_prediction(self):
        autoets = AutoETS(season_length=12, model="AAA")
        fcst_ets = autoets.forecast(ap, 13, None, None, (80, 95), True)

        # test conformal prediction
        autoets_c = AutoETS(
            season_length=12,
            model="AAA",
            prediction_intervals=ConformalIntervals(h=13, n_windows=2),
        )
        assert_class(autoets_c, x=ap, h=13, level=[90, 80], test_forward=True)
        fcst_ets_c = autoets_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_ets_c["mean"], fcst_ets["mean"])
        # _plot_fcst(fcst_ets_c)

    @pytest.mark.parametrize(
        "model",
        [
            "ANNN",
            "AANN",
            "ANAN",
            "AAAN",
            "AAND",
            "AAAD",  # class 1
            "MNNN",
            "MANN",
            "MAND",
            "MNAN",
            "MAAN",
            "MAAD",  # class 2
            "MNMN",
            "MAMN",
            "MAMD",
        ],  # class 3
    )
    def test_models(self, model):
        # Test whether forecast and fit-predict generate the same result
        mod = model[0:3]
        damped_val = model[-1]
        if damped_val == "N":
            damped = False
        else:
            damped = True

        ets = AutoETS(season_length=12, model=mod, damped=damped)
        assert_class(ets, x=ap, h=13, level=[90, 80], test_forward=True)


class TestCES:
    def test_ces(self):
        autoces = AutoCES(season_length=12)
        assert_class(autoces, x=ap, h=12, level=[90, 80], test_forward=True)

    def test_alias_arg(self):
        # test alias argument
        assert repr(AutoCES()) == "CES"
        assert repr(AutoCES(alias="AutoCES_custom")) == "AutoCES_custom"

    def test_ces_conformal_prediction(self):
        autoces = AutoCES(season_length=12)
        fcst_ces = autoces.forecast(ap, 13, None, None, (80, 95), True)

        # test conformal prediction
        autoces_c = AutoCES(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(autoces_c, x=ap, h=13, level=[90, 80], test_forward=True)
        fcst_ces_c = autoces_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_ces["mean"], fcst_ces_c["mean"])

    def test_fit_predict_consistency(self):
        autoces = AutoCES(season_length=12)
        fcst_ces = autoces.forecast(ap, 13, None, None, (80, 95), True)

        fit = autoces.fit(ap)
        fcst = fit.predict(13, None, (80, 95))

        values = ["mean", "lo-80", "lo-95", "hi-80", "hi-95"]
        for k in range(0, len(values)):
            np.testing.assert_equal(fcst_ces[values[k]], fcst[values[k]])

    @pytest.mark.parametrize(
        "values",
        [
            "fitted",
            "fitted-lo-80",
            "fitted-lo-95",
            "fitted-hi-80",
            "fitted-hi-95",
        ],
    )
    def test_in_sample_prediction(self, values):
        autoces = AutoCES(season_length=12)
        fcst_ces = autoces.forecast(ap, 13, None, None, (80, 95), True)

        fit = autoces.fit(ap)
        pi_insample = fit.predict_in_sample((80, 95))

        np.testing.assert_equal(fcst_ces[values], pi_insample[values])


class TestTheta:
    def test_theta(self):
        theta = AutoTheta(season_length=12)
        assert_class(theta, x=ap, h=12, level=[80, 90], test_forward=True)

    def test_alias_arg(self):
        # test alias argument
        assert repr(AutoTheta()) == "AutoTheta"
        assert repr(AutoTheta(alias="AutoTheta_custom")) == "AutoTheta_custom"

    def test_theta_conformal_prediction(self):
        theta = AutoTheta(season_length=12)
        fcst_theta = theta.forecast(ap, 13, None, None, (80, 95), True)

        # test conformal prediction
        theta_c = AutoTheta(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(theta_c, x=ap, h=13, level=[90, 80], test_forward=True)
        fcst_theta_c = theta_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_theta_c["mean"], fcst_theta["mean"])

    def test_zero_theta(self):
        theta = AutoTheta(season_length=12)
        theta.fit(ap)  # Need to fit the model first
        zero_theta = theta.forward(np.zeros(10), h=12, level=[80, 90], fitted=True)
        assert zero_theta is not None


class TestMFLES:
    @classmethod
    def setup_class(cls):
        cls.deg_ts = np.zeros(10)
        cls.h = 12
        cls.X = np.random.rand(ap.size, 2)
        cls.X_future = np.random.rand(cls.h, 2)

    def test_mfles(self):
        auto_mfles = AutoMFLES(test_size=self.h, season_length=12)
        assert_class(
            auto_mfles,
            x=self.deg_ts,
            X=self.X,
            X_future=self.X_future,
            h=self.h,
            skip_insample=False,
            test_forward=False,
        )

    def test_mfles_conformal_prediction(self):
        auto_mfles = AutoMFLES(
            test_size=self.h,
            season_length=12,
            prediction_intervals=ConformalIntervals(h=self.h, n_windows=2),
        )
        assert_class(
            auto_mfles,
            x=ap,
            X=self.X,
            X_future=self.X_future,
            h=self.h,
            skip_insample=False,
            level=[80, 95],
            test_forward=False,
        )
        fcst_auto_mfles = auto_mfles.forecast(
            ap, self.h, X=self.X, X_future=self.X_future, fitted=True, level=[80, 95]
        )
        assert fcst_auto_mfles is not None


class TestTBATS:
    def test_tbats(self):
        tbats = AutoTBATS(season_length=12)
        assert_class(tbats, x=ap, h=12, level=[90, 80])
        fcst_tbats = tbats.forecast(ap, 13, None, None, (80, 95), True)
        assert fcst_tbats is not None


class TestARIMAFamily:
    def test_simple_arima(self):
        simple_arima = ARIMA(order=(1, 0, 0), season_length=12)
        assert_class(simple_arima, x=ap, h=12, level=[90, 80], test_forward=True)

    def test_alias_arg(self):
        # test alias argument
        assert repr(ARIMA()) == "ARIMA"
        assert repr(ARIMA(alias="ARIMA_seasonality")) == "ARIMA_seasonality"

    def test_arima_conformal_prediction(self):
        simple_arima = ARIMA(order=(1, 0, 0), season_length=12)
        fcst_simple_arima = simple_arima.forecast(ap, 13, None, None, (80, 95), True)

        # test conformal prediction
        simple_arima_c = ARIMA(
            order=(1, 0, 0),
            season_length=12,
            prediction_intervals=ConformalIntervals(h=13, n_windows=2),
        )
        assert_class(simple_arima_c, x=ap, h=13, level=[90, 80], test_forward=True)
        fcst_simple_arima_c = simple_arima_c.forecast(
            ap, 13, None, None, (80, 95), True
        )
        np.testing.assert_array_equal(
            fcst_simple_arima_c["mean"], fcst_simple_arima["mean"]
        )

    def test_arima_fixed_params(self):
        simple_arima = ARIMA(
            order=(2, 0, 0), season_length=12, fixed={"ar1": 0.5, "ar2": 0.5}
        )
        assert_class(simple_arima, x=ap, h=12, level=[90, 80], test_forward=True)
        fcst_simple_arima = simple_arima.forecast(ap, 4, None, None, (80, 95), True)
        np.testing.assert_array_equal(
            fcst_simple_arima["mean"], np.array([411.0, 421.5, 416.25, 418.875])
        )

    def test_autoregressive(self):
        ar = AutoRegressive(lags=[12], fixed={"ar12": 0.9999999})
        assert_class(ar, x=ap, h=12, level=[90, 80], test_forward=True)
        fcst_ar = ar.forecast(ap, 13, None, None, (80, 95), True)
        # we should recover seasonal naive
        np.testing.assert_almost_equal(fcst_ar["mean"][:-1], ap[-12:], decimal=4)

    def test_autoregressive_conformal_prediction(self):
        ar = AutoRegressive(lags=[12], fixed={"ar12": 0.9999999})
        fcst_ar = ar.forecast(ap, 13, None, None, (80, 95), True)

        # test conformal prediction
        ar_c = AutoRegressive(
            lags=[12],
            fixed={"ar12": 0.9999999},
            prediction_intervals=ConformalIntervals(h=13, n_windows=2),
        )
        assert_class(ar_c, x=ap, h=13, level=[90, 80], test_forward=True)
        fcst_ar_c = ar_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_ar_c["mean"], fcst_ar["mean"])

    def test_autoregressive_alias_arg(self):
        # test alias argument
        assert repr(AutoRegressive(lags=[12])) == "AutoRegressive"
        assert (
            repr(AutoRegressive(lags=[12], alias="AutoRegressive_lag12"))
            == "AutoRegressive_lag12"
        )

        ar = AutoRegressive(lags=[12], fixed={"ar12": 0.9999999})
        assert_class(ar, x=ap, h=12, level=[90, 80], test_forward=True)
        fcst_ar = ar.forecast(ap, 13, None, None, (80, 95), True)
        # we should recover seasonal naive
        np.testing.assert_almost_equal(fcst_ar["mean"][:-1], ap[-12:], decimal=4)


class TestSES:
    def test_ses(self):
        ses = SimpleExponentialSmoothing(alpha=0.1)
        assert_class(ses, x=ap, h=12)

    def test_alias_arg(self):
        # test alias argument
        assert repr(SimpleExponentialSmoothing(alpha=0.1)) == "SES"
        assert (
            repr(SimpleExponentialSmoothing(alpha=0.1, alias="SES_custom"))
            == "SES_custom"
        )

    def test_ses_fit_predict(self):
        ses = SimpleExponentialSmoothing(alpha=0.1)
        ses = ses.fit(ap)
        fcst_ses = ses.predict(12)
        np.testing.assert_allclose(fcst_ses["mean"], np.repeat(460.3028, 12), rtol=1e-4)

        # to recover these residuals from R
        # you have to pass initial="simple"
        # in the `ses` function
        np.testing.assert_allclose(
            ses.predict_in_sample()["fitted"][[0, 1, -1]],
            np.array([np.nan, 118 - 6.0, 432 + 31.447525]),
        )

    def test_ses_conformal_prediction(self):
        ses = SimpleExponentialSmoothing(alpha=0.1)
        ses = ses.fit(ap)
        fcst_ses = ses.predict(12)

        # test conformal prediction
        ses_c = SimpleExponentialSmoothing(
            alpha=0.1, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(
            ses_c, x=ap, h=13, level=[90, 80], test_forward=False, skip_insample=True
        )
        fcst_ses_c = ses_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_ses_c["mean"][:12], fcst_ses["mean"])

    def test_ses_optimized(self):
        ses_op = SimpleExponentialSmoothingOptimized()
        assert_class(ses_op, x=ap, h=12)
        ses_op = ses_op.fit(ap)
        fcst_ses_op = ses_op.predict(12)
        assert fcst_ses_op is not None

    def test_ses_optimized_conformal_prediction(self):
        ses_op = SimpleExponentialSmoothingOptimized()
        ses_op = ses_op.fit(ap)
        fcst_ses_op = ses_op.predict(12)

        # test conformal prediction
        ses_op_c = SimpleExponentialSmoothingOptimized(
            prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(ses_op_c, x=ap, h=13, level=[90, 80], skip_insample=True)
        fcst_ses_op_c = ses_op_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_ses_op_c["mean"][:12], fcst_ses_op["mean"])

    def test_ses_optimized_alias_arg(self):
        # test alias argument
        assert repr(SimpleExponentialSmoothingOptimized()) == "SESOpt"
        assert (
            repr(SimpleExponentialSmoothingOptimized(alias="SESOpt_custom"))
            == "SESOpt_custom"
        )


class TestSeasonalES:
    def test_seasonal_es(self):
        seas_es = SeasonalExponentialSmoothing(season_length=12, alpha=1.0)
        assert_class(seas_es, x=ap, h=12)
        np.testing.assert_array_equal(
            seas_es.predict_in_sample()["fitted"][-3:],
            np.array([461 - 54.0, 390 - 28.0, 432 - 27.0]),
        )

    def test_alias_arg(self):
        # test alias argument
        assert (
            repr(SeasonalExponentialSmoothing(season_length=12, alpha=1.0))
            == "SeasonalES"
        )
        assert (
            repr(
                SeasonalExponentialSmoothing(
                    season_length=12, alpha=1.0, alias="SeasonalES_custom"
                )
            )
            == "SeasonalES_custom"
        )

    def test_seasonal_es_conformal_prediction(self):
        seas_es = SeasonalExponentialSmoothing(season_length=12, alpha=1.0)
        seas_es = seas_es.fit(ap)
        fcst_seas_es = seas_es.predict(12)

        # test conformal prediction
        seas_es_c = SeasonalExponentialSmoothing(
            season_length=12,
            alpha=1.0,
            prediction_intervals=ConformalIntervals(h=13, n_windows=2),
        )
        assert_class(
            seas_es_c,
            x=ap,
            h=13,
            level=[90, 80],
            test_forward=False,
            skip_insample=True,
        )
        fcst_seas_es_c = seas_es_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_seas_es_c["mean"][:12], fcst_seas_es["mean"])

    def test_seasonal_es_seasonality_recovery(self):
        seas_es = SeasonalExponentialSmoothing(season_length=12, alpha=1.0)
        seas_es = seas_es.fit(ap)

        # test we can recover the expected seasonality
        np.testing.assert_array_equal(
            seas_es.forecast(ap[4:], h=12)["mean"], seas_es.forecast(ap, h=12)["mean"]
        )

    def test_seasonal_es_close_to_naive(self):
        seas_es = SeasonalExponentialSmoothing(season_length=12, alpha=1.0)
        seas_es = seas_es.fit(ap)

        # test close to seasonal naive
        for i in range(1, 13):
            np.testing.assert_allclose(
                ap[i:][-12:], seas_es.forecast(ap[i:], h=12)["mean"], rtol=1e-1
            )

    def test_seasonal_es_optimized(self):
        seas_es_opt = SeasonalExponentialSmoothingOptimized(season_length=12)
        assert_class(seas_es_opt, x=ap, h=12)
        fcst_seas_es_opt = seas_es_opt.forecast(ap, h=12)
        assert fcst_seas_es_opt is not None

    def test_seasonal_es_optimized_conformal_prediction(self):
        seas_es_opt = SeasonalExponentialSmoothingOptimized(season_length=12)
        fcst_seas_es_opt = seas_es_opt.forecast(ap, h=12)

        # test conformal prediction
        seas_es_opt_c = SeasonalExponentialSmoothingOptimized(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(
            seas_es_opt_c,
            x=ap,
            h=13,
            level=[90, 80],
            test_forward=False,
            skip_insample=True,
        )
        fcst_seas_es_opt_c = seas_es_opt_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(
            fcst_seas_es_opt_c["mean"][:12], fcst_seas_es_opt["mean"]
        )

    def test_seasonal_es_optimized_close_to_naive(self):
        seas_es_opt = SeasonalExponentialSmoothingOptimized(season_length=12)
        for i in range(1, 13):
            np.testing.assert_allclose(
                ap[i:][-12:], seas_es_opt.forecast(ap[i:], h=12)["mean"], rtol=0.8
            )

    def test_seasonal_es_optimized_alias_arg(self):
        # test alias argument
        assert (
            repr(SeasonalExponentialSmoothingOptimized(season_length=12)) == "SeasESOpt"
        )
        assert (
            repr(
                SeasonalExponentialSmoothingOptimized(
                    season_length=12, alias="SeasESOpt_custom"
                )
            )
            == "SeasESOpt_custom"
        )


class TestHolt:
    def test_holt_equivalence_with_ets(self):
        holt = Holt(season_length=12, error_type="A")
        fcast_holt = holt.forecast(ap, 12)

        ets = AutoETS(season_length=12, model="AAN")
        fcast_ets = ets.forecast(ap, 12)

        np.testing.assert_equal(fcast_holt, fcast_ets)

    def test_holt_fit_predict_equivalence_with_ets(self):
        holt = Holt(season_length=12, error_type="A")
        holt.fit(ap)
        fcast_holt = holt.predict(12)

        ets = AutoETS(season_length=12, model="AAN")
        fcast_ets = ets.forecast(ap, 12)

        np.testing.assert_equal(fcast_holt, fcast_ets)

    def test_holt_conformal_prediction_equivalence_with_ets(self):
        holt_c = Holt(
            season_length=12,
            error_type="A",
            prediction_intervals=ConformalIntervals(h=12, n_windows=2),
        )
        fcast_holt_c = holt_c.forecast(ap, 12, level=[80, 90])

        ets_c = AutoETS(
            season_length=12,
            model="AAN",
            prediction_intervals=ConformalIntervals(h=12, n_windows=2),
        )
        fcast_ets_c = ets_c.forecast(ap, 12, level=[80, 90])

        np.testing.assert_equal(fcast_holt_c, fcast_ets_c)

    def test_alias_arg(self):
        # test alias argument
        assert repr(Holt()) == "Holt"
        assert repr(Holt(alias="Holt_custom")) == "Holt_custom"


class TestHoltWinters:
    def test_holt_winters_equivalence_with_ets(self):
        hw = HoltWinters(season_length=12, error_type="A")
        fcast_hw = hw.forecast(ap, 12)

        ets = AutoETS(season_length=12, model="AAA")
        fcast_ets = ets.forecast(ap, 12)

        np.testing.assert_equal(fcast_hw, fcast_ets)

    def test_holt_winters_conformal_prediction_equivalence_with_ets(self):
        hw_c = HoltWinters(
            season_length=12,
            error_type="A",
            prediction_intervals=ConformalIntervals(h=12, n_windows=2),
        )
        fcast_hw_c = hw_c.forecast(ap, 12, level=[80, 90])

        ets_c = AutoETS(
            season_length=12,
            model="AAA",
            prediction_intervals=ConformalIntervals(h=12, n_windows=2),
        )
        fcast_ets_c = ets_c.forecast(ap, 12, level=[80, 90])

        np.testing.assert_equal(fcast_hw_c, fcast_ets_c)

    def test_holt_winters_fit_predict_equivalence_with_ets(self):
        hw = HoltWinters(season_length=12, error_type="A")
        hw.fit(ap)
        fcast_hw = hw.predict(12)

        ets = AutoETS(season_length=12, model="AAA")
        fcast_ets = ets.forecast(ap, 12)

        np.testing.assert_equal(fcast_hw, fcast_ets)

    def test_alias_arg(self):
        # test alias argument
        assert repr(HoltWinters()) == "HoltWinters"
        assert repr(HoltWinters(alias="HoltWinters_custom")) == "HoltWinters_custom"


class TestHistoricAverage:
    def test_historic_average(self):
        ha = HistoricAverage()
        assert_class(ha, x=ap, h=12, level=[80, 90])

    def test_historic_average_fit_predict(self):
        ha = HistoricAverage()
        ha.fit(ap)
        fcst_ha = ha.predict(12)
        np.testing.assert_allclose(fcst_ha["mean"], np.repeat(ap.mean(), 12), rtol=1e-5)

        np.testing.assert_almost_equal(
            ha.predict_in_sample()["fitted"][:4],
            np.array([280.2986, 280.2986, 280.2986, 280.2986]),
            decimal=4,
        )

    def test_historic_average_forecast(self):
        ha = HistoricAverage()
        fcst_ha = ha.forecast(ap, 12, None, None, (80, 95), True)
        np.testing.assert_almost_equal(
            fcst_ha["lo-80"], np.repeat(126.0227, 12), decimal=4
        )

    def test_historic_average_conformal_prediction(self):
        ha = HistoricAverage()
        fcst_ha = ha.forecast(ap, 12, None, None, (80, 95), True)

        # test conformal prediction
        ha_c = HistoricAverage(
            prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(ha_c, x=ap, h=13, level=[90, 80], test_forward=False)
        fcst_ha_c = ha_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_ha_c["mean"][:12], fcst_ha["mean"])

    def test_alias_arg(self):
        # test alias argument
        assert repr(HistoricAverage()) == "HistoricAverage"
        assert (
            repr(HistoricAverage(alias="HistoricAverage_custom"))
            == "HistoricAverage_custom"
        )


class TestNaive:
    def test_naive_prediction_intervals(self):
        # Test prediction intervals - forecast
        naive = Naive()
        naive.forecast(ap, 12)
        naive.forecast(ap, 12, None, None, (80, 95), True)

        # Test prediction intervals - fit & predict
        naive.fit(ap)
        naive.predict(12)
        naive.predict(12, None, (80, 95))

    def test_naive(self):
        naive = Naive()
        assert_class(naive, x=ap, h=12, level=[90, 80])
        naive.fit(ap)
        fcst_naive = naive.predict(12)
        np.testing.assert_allclose(fcst_naive["mean"], np.repeat(ap[-1], 12), rtol=1e-5)

    def test_naive_forecast(self):
        naive = Naive()
        fcst_naive = naive.forecast(ap, 12, None, None, (80, 95), True)
        np.testing.assert_almost_equal(
            fcst_naive["lo-80"],
            np.array(
                [
                    388.7984,
                    370.9037,
                    357.1726,
                    345.5967,
                    335.3982,
                    326.1781,
                    317.6992,
                    309.8073,
                    302.3951,
                    295.3845,
                    288.7164,
                    282.3452,
                ]
            ),
            decimal=4,
        )

    def test_naive_forward(self):
        # Unit test forward:=forecast
        naive = Naive()
        fcst_naive = naive.forward(ap, 12, None, None, (80, 95), True)
        np.testing.assert_almost_equal(
            fcst_naive["lo-80"],
            np.array(
                [
                    388.7984,
                    370.9037,
                    357.1726,
                    345.5967,
                    335.3982,
                    326.1781,
                    317.6992,
                    309.8073,
                    302.3951,
                    295.3845,
                    288.7164,
                    282.3452,
                ]
            ),
            decimal=4,
        )

    def test_naive_conformal_prediction(self):
        naive = Naive()
        fcst_naive = naive.forecast(ap, 12, None, None, (80, 95), True)

        # test conformal prediction
        naive_c = Naive(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
        assert_class(naive_c, x=ap, h=13, level=[90, 80], test_forward=False)
        fcst_naive_c = naive_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_naive_c["mean"][:12], fcst_naive["mean"])

    def test_alias_arg(self):
        # test alias argument
        assert repr(Naive()) == "Naive"
        assert repr(Naive(alias="Naive_custom")) == "Naive_custom"


class TestRandomWalkWithDrift:
    def test_rwd_prediction_intervals(self):
        # Test prediction intervals - forecast
        rwd = RandomWalkWithDrift()
        rwd.forecast(ap, 12)
        rwd.forecast(ap, 12, None, None, (80, 95), True)

        # Test prediction intervals - fit & predict
        rwd = RandomWalkWithDrift()
        rwd.fit(ap)
        rwd.predict(12)
        rwd.predict(12, None, (80, 95))

    def test_rwd(self):
        rwd = RandomWalkWithDrift()
        assert_class(rwd, x=ap, h=12, level=[90, 80])
        rwd = rwd.fit(ap)
        fcst_rwd = rwd.predict(12)
        np.testing.assert_allclose(
            fcst_rwd["mean"][:2], np.array([434.2378, 436.4755]), rtol=1e-4
        )

        np.testing.assert_almost_equal(
            rwd.predict_in_sample()["fitted"][:3],
            np.array([np.nan, 118 - 3.7622378, 132 - 11.7622378]),
            decimal=6,
        )

    def test_rwd_conformal_prediction(self):
        rwd = RandomWalkWithDrift()
        rwd = rwd.fit(ap)
        fcst_rwd = rwd.predict(12)

        # test conformal prediction
        rwd_c = RandomWalkWithDrift(
            prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(rwd_c, x=ap, h=13, level=[90, 80], test_forward=False)
        fcst_rwd_c = rwd_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_rwd_c["mean"][:12], fcst_rwd["mean"])

    def test_rwd_forecast(self):
        rwd = RandomWalkWithDrift()
        fcst_rwd = rwd.forecast(
            y=ap, h=12, X=None, X_future=None, level=(80, 95), fitted=True
        )
        np.testing.assert_almost_equal(
            fcst_rwd["lo-80"],
            np.array(
                [
                    390.9799,
                    375.0862,
                    363.2664,
                    353.5325,
                    345.1178,
                    337.6304,
                    330.8384,
                    324.5916,
                    318.7857,
                    313.3453,
                    308.2136,
                    303.3469,
                ]
            ),
            decimal=1,
        )

    def test_alias_arg(self):
        # test alias argument
        assert repr(RandomWalkWithDrift()) == "RWD"
        assert repr(RandomWalkWithDrift(alias="RWD_custom")) == "RWD_custom"


class TestSeasonalNaive:
    def test_seasonal_naive_prediction_intervals(self):
        # Test prediction intervals - forecast
        seas_naive = SeasonalNaive(season_length=12)
        seas_naive.forecast(ap, 12)
        seas_naive.forecast(ap, 12, None, None, (80, 95), True)

        # Test prediction intervals - fit and predict
        seas_naive = SeasonalNaive(season_length=12)
        seas_naive.fit(ap)
        seas_naive.predict(12)
        seas_naive.predict(12, None, (80, 95))

    def test_seasonal_naive(self):
        seas_naive = SeasonalNaive(season_length=12)
        assert_class(seas_naive, x=ap, h=12, level=[90, 80])
        seas_naive = seas_naive.fit(ap)
        fcst_seas_naive = seas_naive.predict(12)
        assert fcst_seas_naive is not None
        np.testing.assert_array_equal(
            seas_naive.predict_in_sample()["fitted"][-3:],
            np.array([461 - 54.0, 390 - 28.0, 432 - 27.0]),
        )

    def test_seasonal_naive_forecast(self):
        seas_naive = SeasonalNaive(season_length=12)
        fcst_seas_naive = seas_naive.forecast(
            y=ap, h=12, X=None, X_future=None, level=(80, 95), fitted=True
        )
        np.testing.assert_almost_equal(
            fcst_seas_naive["lo-80"],
            np.array(
                [
                    370.4595,
                    344.4595,
                    372.4595,
                    414.4595,
                    425.4595,
                    488.4595,
                    575.4595,
                    559.4595,
                    461.4595,
                    414.4595,
                    343.4595,
                    385.4595,
                ]
            ),
            decimal=4,
        )

    def test_seasonal_naive_forward(self):
        # Unit test forward:=forecast
        seas_naive = SeasonalNaive(season_length=12)
        fcst_seas_naive = seas_naive.forward(
            y=ap, h=12, X=None, X_future=None, level=(80, 95), fitted=True
        )
        np.testing.assert_almost_equal(
            fcst_seas_naive["lo-80"],
            np.array(
                [
                    370.4595,
                    344.4595,
                    372.4595,
                    414.4595,
                    425.4595,
                    488.4595,
                    575.4595,
                    559.4595,
                    461.4595,
                    414.4595,
                    343.4595,
                    385.4595,
                ]
            ),
            decimal=4,
        )

    def test_seasonal_naive_big_h(self):
        # test h > season_length
        seas_naive_bigh = SeasonalNaive(season_length=12)
        fcst_seas_naive_bigh = seas_naive_bigh.forecast(
            y=ap, h=13, X=None, X_future=None, level=(80, 95), fitted=True
        )
        np.testing.assert_almost_equal(
            fcst_seas_naive_bigh["lo-80"][:12],
            np.array(
                [
                    370.4595,
                    344.4595,
                    372.4595,
                    414.4595,
                    425.4595,
                    488.4595,
                    575.4595,
                    559.4595,
                    461.4595,
                    414.4595,
                    343.4595,
                    385.4595,
                ]
            ),
            decimal=4,
        )

    def test_seasonal_naive_conformal_prediction(self):
        seas_naive = SeasonalNaive(season_length=12)
        fcst_seas_naive = seas_naive.forecast(
            y=ap, h=12, X=None, X_future=None, level=(80, 95), fitted=True
        )

        # test conformal prediction
        seas_naive_c = SeasonalNaive(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(seas_naive_c, x=ap, h=13, level=[90, 80], test_forward=False)
        fcst_seas_naive_c = seas_naive_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(
            fcst_seas_naive_c["mean"][:12], fcst_seas_naive["mean"]
        )

    def test_alias_arg(self):
        # test alias argument
        assert repr(SeasonalNaive(12)) == "SeasonalNaive"
        assert (
            repr(SeasonalNaive(12, alias="SeasonalNaive_custom"))
            == "SeasonalNaive_custom"
        )


class TestWindowAverage:
    def test_window_average(self):
        w_avg = WindowAverage(window_size=24)
        assert_class(w_avg, x=ap, h=12, skip_insample=True)
        w_avg = w_avg.fit(ap)
        fcst_w_avg = w_avg.predict(12)
        np.testing.assert_allclose(fcst_w_avg["mean"], np.repeat(ap[-24:].mean(), 12))

    def test_window_average_conformal_prediction(self):
        w_avg = WindowAverage(window_size=24)
        w_avg = w_avg.fit(ap)
        fcst_w_avg = w_avg.predict(12)

        # test conformal prediction
        w_avg_c = WindowAverage(
            window_size=24, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(
            w_avg_c, x=ap, h=13, level=[90, 80], test_forward=False, skip_insample=True
        )
        fcst_w_avg_c = w_avg_c.forecast(ap, 13, None, None, (80, 95), False)
        np.testing.assert_array_equal(fcst_w_avg_c["mean"][:12], fcst_w_avg["mean"])

    def test_alias_arg(self):
        # test alias argument
        assert repr(WindowAverage(1)) == "WindowAverage"
        assert (
            repr(WindowAverage(1, alias="WindowAverage_custom"))
            == "WindowAverage_custom"
        )


class TestSeasonalWindowAverage:
    def test_seasonal_window_average(self):
        seas_w_avg = SeasonalWindowAverage(season_length=12, window_size=1)
        assert_class(seas_w_avg, x=ap, h=12, skip_insample=True)
        seas_w_avg = seas_w_avg.fit(ap)
        fcst_seas_w_avg = seas_w_avg.predict(12)
        assert fcst_seas_w_avg is not None

    def test_seasonal_window_average_conformal_prediction(self):
        seas_w_avg = SeasonalWindowAverage(season_length=12, window_size=1)
        seas_w_avg = seas_w_avg.fit(ap)
        fcst_seas_w_avg = seas_w_avg.predict(12)

        # test conformal prediction
        seas_w_avg_c = SeasonalWindowAverage(
            season_length=12,
            window_size=1,
            prediction_intervals=ConformalIntervals(h=13, n_windows=2),
        )
        assert_class(
            seas_w_avg_c,
            x=ap,
            h=13,
            level=[90, 80],
            test_forward=False,
            skip_insample=True,
        )
        fcst_seas_w_avg_c = seas_w_avg_c.forecast(ap, 13, None, None, (80, 95), False)
        np.testing.assert_array_equal(
            fcst_seas_w_avg_c["mean"][:12], fcst_seas_w_avg["mean"]
        )

    def test_alias_arg(self):
        # test alias argument
        assert repr(SeasonalWindowAverage(12, 1)) == "SeasWA"
        assert (
            repr(SeasonalWindowAverage(12, 1, alias="SeasWA_custom")) == "SeasWA_custom"
        )


class TestADIDA:
    @classmethod
    def setup_class(cls):
        cls.deg_ts = np.zeros(10)

    def test_adida(self):
        adida = ADIDA()
        assert_class(adida, x=ap, h=12, skip_insample=False)
        assert_class(adida, x=self.deg_ts, h=12, skip_insample=False)
        fcst_adida = adida.forecast(ap, 12)
        assert fcst_adida is not None

    def test_adida_conformal_prediction(self):
        adida = ADIDA()
        fcst_adida = adida.forecast(ap, 12)

        # test conformal prediction
        adida_c = ADIDA(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
        assert_class(adida_c, x=ap, h=13, level=[90, 80], skip_insample=False)
        fcst_adida_c = adida_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_adida_c["mean"][:12], fcst_adida["mean"])

    def test_alias_arg(self):
        # test alias argument
        assert repr(ADIDA()) == "ADIDA"
        assert repr(ADIDA(alias="ADIDA_custom")) == "ADIDA_custom"


class TestCrostonClassic:
    @classmethod
    def setup_class(cls):
        cls.deg_ts = np.zeros(10)

    def test_croston_classic(self):
        croston = CrostonClassic(prediction_intervals=ConformalIntervals(2, 1))
        assert_class(croston, x=ap, h=12, skip_insample=False, level=[80])
        assert_class(croston, x=self.deg_ts, h=12, skip_insample=False, level=[80])
        fcst_croston = croston.forecast(ap, 12)
        assert fcst_croston is not None

    def test_croston_classic_conformal_prediction(self):
        croston = CrostonClassic(prediction_intervals=ConformalIntervals(2, 1))
        fcst_croston = croston.forecast(ap, 12)

        # test conformal prediction
        croston_c = CrostonClassic(
            prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(croston_c, x=ap, h=13, level=[90.0, 80.0], skip_insample=False)
        fcst_croston_c = croston_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_croston_c["mean"][:12], fcst_croston["mean"])

    def test_alias_arg(self):
        # test alias argument
        assert repr(CrostonClassic()) == "CrostonClassic"
        assert (
            repr(CrostonClassic(alias="CrostonClassic_custom"))
            == "CrostonClassic_custom"
        )


class TestCrostonOptimized:
    @classmethod
    def setup_class(cls):
        cls.deg_ts = np.zeros(10)

    def test_croston_optimized(self):
        croston_op = CrostonOptimized(prediction_intervals=ConformalIntervals(2, 1))
        assert_class(croston_op, x=ap, h=12, skip_insample=False, level=[80])
        assert_class(croston_op, x=self.deg_ts, h=12, skip_insample=False, level=[80])
        fcst_croston_op = croston_op.forecast(ap, 12)
        assert fcst_croston_op is not None

    def test_croston_optimized_conformal_prediction(self):
        croston_op = CrostonOptimized(prediction_intervals=ConformalIntervals(2, 1))
        fcst_croston_op = croston_op.forecast(ap, 12)

        # test conformal prediction
        croston_op_c = CrostonOptimized(
            prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(croston_op_c, x=ap, h=13, level=[90, 80], skip_insample=False)
        fcst_croston_op_c = croston_op_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(
            fcst_croston_op_c["mean"][:12], fcst_croston_op["mean"]
        )

    def test_alias_arg(self):
        # test alias argument
        assert repr(CrostonOptimized()) == "CrostonOptimized"
        assert (
            repr(CrostonOptimized(alias="CrostonOptimized_custom"))
            == "CrostonOptimized_custom"
        )


class TestCrostonSBA:
    @classmethod
    def setup_class(cls):
        cls.deg_ts = np.zeros(10)

    def test_croston_sba(self):
        croston_sba = CrostonSBA(prediction_intervals=ConformalIntervals(2, 1))
        assert_class(croston_sba, x=ap, h=12, skip_insample=False, level=[80])
        assert_class(croston_sba, x=self.deg_ts, h=12, skip_insample=False, level=[80])
        fcst_croston_sba = croston_sba.forecast(ap, 12)
        assert fcst_croston_sba is not None

    def test_croston_sba_conformal_prediction(self):
        croston_sba = CrostonSBA(prediction_intervals=ConformalIntervals(2, 1))
        fcst_croston_sba = croston_sba.forecast(ap, 12)

        # test conformal prediction
        croston_sba_c = CrostonSBA(
            prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(croston_sba_c, x=ap, h=13, level=[90, 80], skip_insample=False)
        fcst_croston_sba_c = croston_sba_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(
            fcst_croston_sba_c["mean"][:12], fcst_croston_sba["mean"]
        )

    def test_alias_arg(self):
        # test alias argument
        assert repr(CrostonSBA()) == "CrostonSBA"
        assert repr(CrostonSBA(alias="CrostonSBA_custom")) == "CrostonSBA_custom"


class TestIMAPA:
    @classmethod
    def setup_class(cls):
        cls.deg_ts = np.zeros(10)

    def test_imapa(self):
        imapa = IMAPA(prediction_intervals=ConformalIntervals(2, 1))
        assert_class(imapa, x=ap, h=12, skip_insample=False, level=[80])
        assert_class(imapa, x=self.deg_ts, h=12, skip_insample=False, level=[80])
        fcst_imapa = imapa.forecast(ap, 12)
        assert fcst_imapa is not None

    def test_imapa_conformal_prediction(self):
        imapa = IMAPA(prediction_intervals=ConformalIntervals(2, 1))
        fcst_imapa = imapa.forecast(ap, 12)

        # test conformal prediction
        imapa_c = IMAPA(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
        assert_class(imapa_c, x=ap, h=13, level=[90, 80], skip_insample=False)
        fcst_imapa_c = imapa_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_imapa_c["mean"][:12], fcst_imapa["mean"])

    def test_alias_arg(self):
        # test alias argument
        assert repr(IMAPA()) == "IMAPA"
        assert repr(IMAPA(alias="IMAPA_custom")) == "IMAPA_custom"


class TestTSB:
    @classmethod
    def setup_class(cls):
        cls.deg_ts = np.zeros(10)

    def test_tsb(self):
        tsb = TSB(
            alpha_d=0.9, alpha_p=0.1, prediction_intervals=ConformalIntervals(2, 1)
        )
        assert_class(tsb, x=ap, h=12, skip_insample=False, level=[80])
        assert_class(tsb, x=self.deg_ts, h=12, skip_insample=False, level=[80])
        fcst_tsb = tsb.forecast(ap, 12)
        assert fcst_tsb is not None

    def test_tsb_conformal_prediction(self):
        tsb = TSB(
            alpha_d=0.9, alpha_p=0.1, prediction_intervals=ConformalIntervals(2, 1)
        )
        fcst_tsb = tsb.forecast(ap, 12)

        # test conformal prediction
        tsb_c = TSB(
            alpha_d=0.9,
            alpha_p=0.1,
            prediction_intervals=ConformalIntervals(h=13, n_windows=2),
        )
        assert_class(tsb_c, x=ap, h=13, level=[90, 80], skip_insample=True)
        fcst_tsb_c = tsb_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_tsb_c["mean"][:12], fcst_tsb["mean"])

    def test_alias_arg(self):
        # test alias argument
        assert repr(TSB(0.9, 0.1)) == "TSB"
        assert repr(TSB(0.9, 0.1, alias="TSB_custom")) == "TSB_custom"


class TestMSTL:
    @pytest.mark.parametrize(
        "trend_forecaster,skip_insample,test_forward",
        [
            (AutoARIMA(), False, False),
            (AutoCES(), True, True),
            (AutoETS(model="ZZN"), False, True),
            (Naive(), False, False),
            (CrostonClassic(), True, False),
        ],
    )
    @pytest.mark.parametrize("stl_kwargs", [None, dict(trend=25)])
    def test_mstl_with_trend_forecasters(
        self, trend_forecaster, skip_insample, test_forward, stl_kwargs
    ):
        mstl_model = MSTL(
            season_length=[12, 14],
            trend_forecaster=trend_forecaster,
            stl_kwargs=stl_kwargs,
        )
        assert_class(
            mstl_model,
            x=ap,
            h=12,
            skip_insample=skip_insample,
            level=None,
            test_forward=test_forward,
        )

    def test_mstl_intervals_native_vs_conformal(self):
        # intervals with & without conformal
        # trend fcst supports level, use native levels
        mstl_native = MSTL(season_length=12, trend_forecaster=ARIMA(order=(0, 1, 0)))
        res_native_fp = pd.DataFrame(
            mstl_native.fit(y=ap).predict(h=24, level=[80, 95])
        )
        res_native_fc = pd.DataFrame(mstl_native.forecast(y=ap, h=24, level=[80, 95]))
        pd.testing.assert_frame_equal(res_native_fp, res_native_fc)

        # trend fcst supports level, use conformal
        mstl_conformal = MSTL(
            season_length=12,
            trend_forecaster=ARIMA(
                order=(0, 1, 0),
                prediction_intervals=ConformalIntervals(h=24),
            ),
        )
        res_conformal_fp = pd.DataFrame(
            mstl_conformal.fit(y=ap).predict(h=24, level=[80, 95])
        )
        res_conformal_fc = pd.DataFrame(
            mstl_conformal.forecast(y=ap, h=24, level=[80, 95])
        )
        pd.testing.assert_frame_equal(res_conformal_fp, res_conformal_fc)

    def test_mstl_trend_forecaster_no_level_support(self):
        # trend fcst doesn't support level
        mstl_bad = MSTL(season_length=12, trend_forecaster=CrostonClassic())
        with pytest.raises(Exception, match="prediction_intervals"):
            mstl_bad.fit(y=ap).predict(h=24, level=[80, 95])
        with pytest.raises(Exception, match="prediction_intervals"):
            mstl_bad.forecast(y=ap, h=24, level=[80, 95])

    @pytest.mark.parametrize(
        "trend_forecaster,skip_insample,test_forward",
        [
            (
                AutoARIMA(prediction_intervals=ConformalIntervals(h=13, n_windows=2)),
                False,
                False,
            ),
            (
                AutoCES(prediction_intervals=ConformalIntervals(h=13, n_windows=2)),
                True,
                True,
            ),
        ],
    )
    @pytest.mark.parametrize("stl_kwargs", [None, dict(trend=25)])
    def test_mstl_conformal_prediction_in_trend_forecaster(
        self, trend_forecaster, skip_insample, test_forward, stl_kwargs
    ):
        mstl_model = MSTL(
            season_length=[12, 14],
            trend_forecaster=trend_forecaster,
            stl_kwargs=stl_kwargs,
        )
        assert_class(
            mstl_model,
            x=ap,
            h=13,
            skip_insample=skip_insample,
            level=[80, 90] if not skip_insample else None,
            test_forward=test_forward,
        )

    @pytest.mark.parametrize("stl_kwargs", [None, dict(trend=25)])
    def test_mstl_conformal_prediction_in_mstl(self, stl_kwargs):
        # conformal prediction
        # define prediction_interval in MSTL
        trend_forecaster = AutoCES()
        skip_insample = False

        mstl_model = MSTL(
            season_length=[12, 14],
            trend_forecaster=trend_forecaster,
            stl_kwargs=stl_kwargs,
            prediction_intervals=ConformalIntervals(h=13, n_windows=2),
        )
        assert_class(
            mstl_model,
            x=ap,
            h=13,
            skip_insample=False,
            level=[80, 90] if not skip_insample else None,
            test_forward=True,
        )

    def test_mstl_seasonal_trend_forecaster_failure(self):
        # fail with seasonal trend forecasters
        with pytest.raises(Exception, match="should not adjust seasonal"):
            MSTL([3, 12], AutoETS(model="ZZZ"))
        with pytest.raises(Exception, match="should not adjust seasonal"):
            MSTL([3, 12], AutoARIMA(season_length=12))

    def test_alias_arg(self):
        # test alias argument
        assert repr(MSTL(season_length=7)) == "MSTL"
        assert repr(MSTL(season_length=7, alias="MSTL_custom")) == "MSTL_custom"


class TestOptimizedTheta:
    def test_optimized_theta_vs_autotheta_otm(self):
        otm = OptimizedTheta(season_length=12)
        fcast_otm = otm.forecast(ap, 12)
        otm.fit(ap)
        forward_otm = otm.forward(y=ap, h=12)

        theta = AutoTheta(season_length=12, model="OTM")
        fcast_theta = theta.forecast(ap, 12)
        theta.fit(ap)
        forward_autotheta = theta.forward(y=ap, h=12)

        np.testing.assert_equal(fcast_theta, fcast_otm)
        np.testing.assert_equal(forward_autotheta, forward_otm)

    def test_optimized_theta_conformal_prediction(self):
        otm_c = OptimizedTheta(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=5)
        )
        assert_class(otm_c, x=ap, h=13, level=[90, 80], test_forward=True)

        otm = OptimizedTheta(season_length=12)
        fcast_otm = otm.forecast(ap, 12)

        fcst_otm_c = otm_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_otm_c["mean"][:12], fcast_otm["mean"])

    def test_optimized_theta_fit_predict_vs_forecast(self):
        otm = OptimizedTheta(season_length=12)
        otm.fit(ap)
        fcast_otm = otm.predict(12)

        theta = AutoTheta(season_length=12, model="OTM")
        fcast_theta = theta.forecast(ap, 12)

        np.testing.assert_equal(fcast_theta, fcast_otm)

    def test_alias_arg(self):
        assert repr(OptimizedTheta()) == "OptimizedTheta"
        assert (
            repr(OptimizedTheta(alias="OptimizedTheta_custom"))
            == "OptimizedTheta_custom"
        )


class TestDynamicTheta:
    def test_dynamic_theta_vs_autotheta_dstm(self):
        dstm = DynamicTheta(season_length=12)
        fcast_dstm = dstm.forecast(ap, 12)
        dstm.fit(ap)
        forward_dstm = dstm.forward(y=ap, h=12)

        theta = AutoTheta(season_length=12, model="DSTM")
        fcast_theta = theta.forecast(ap, 12)
        theta.fit(ap)
        forward_autotheta = theta.forward(y=ap, h=12)

        np.testing.assert_equal(fcast_theta, fcast_dstm)
        np.testing.assert_equal(forward_autotheta, forward_dstm)

    def test_dynamic_theta_fit_predict_vs_forecast(self):
        dstm = DynamicTheta(season_length=12)
        dstm.fit(ap)
        fcast_dstm = dstm.predict(12)

        theta = AutoTheta(season_length=12, model="DSTM")
        fcast_theta = theta.forecast(ap, 12)

        np.testing.assert_equal(fcast_theta, fcast_dstm)

    def test_dynamic_theta_conformal_prediction(self):
        dstm_c = DynamicTheta(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=5)
        )
        assert_class(dstm_c, x=ap, h=13, level=[90, 80], test_forward=True)

        dstm = DynamicTheta(season_length=12)
        fcast_dstm = dstm.forecast(ap, 12)

        fcst_dstm_c = dstm_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_dstm_c["mean"][:12], fcast_dstm["mean"])

    def test_alias_arg(self):
        assert repr(DynamicTheta()) == "DynamicTheta"
        assert repr(DynamicTheta(alias="DynamicTheta_custom")) == "DynamicTheta_custom"


class TestDynamicOptimizedTheta:
    def test_dynamic_optimized_theta_vs_autotheta_dotm(self):
        dotm = DynamicOptimizedTheta(season_length=12)
        fcast_dotm = dotm.forecast(ap, 12)
        dotm.fit(ap)
        forward_dotm = dotm.forward(y=ap, h=12)

        theta = AutoTheta(season_length=12, model="DOTM")
        fcast_theta = theta.forecast(ap, 12)
        theta.fit(ap)
        forward_autotheta = theta.forward(y=ap, h=12)

        np.testing.assert_equal(fcast_theta, fcast_dotm)
        np.testing.assert_equal(forward_autotheta, forward_dotm)

    def test_dynamic_optimized_theta_fit_predict_vs_forecast(self):
        dotm = DynamicOptimizedTheta(season_length=12)
        dotm.fit(ap)
        fcast_dotm = dotm.predict(12)

        theta = AutoTheta(season_length=12, model="DOTM")
        fcast_theta = theta.forecast(ap, 12)

        np.testing.assert_equal(fcast_theta, fcast_dotm)

    def test_dynamic_optimized_theta_conformal_prediction(self):
        dotm_c = DynamicOptimizedTheta(
            season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=5)
        )
        assert_class(dotm_c, x=ap, h=13, level=[90, 80], test_forward=True)

        dotm = DynamicOptimizedTheta(season_length=12)
        fcast_dotm = dotm.forecast(ap, 12)

        fcst_dotm_c = dotm_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_dotm_c["mean"][:12], fcast_dotm["mean"])

    def test_alias_arg(self):
        assert repr(DynamicOptimizedTheta()) == "DynamicOptimizedTheta"
        assert (
            repr(DynamicOptimizedTheta(alias="DynamicOptimizedTheta_custom"))
            == "DynamicOptimizedTheta_custom"
        )


class TestGARCH:
    @classmethod
    def setup_class(cls):
        # Generate GARCH(2,2) data
        n = 1000
        w = 0.5
        alpha = np.array([0.1, 0.2])
        beta = np.array([0.4, 0.2])
        cls.y = generate_garch_data(n, w, alpha, beta)

    def test_garch_basic(self):
        garch = GARCH(2, 2)
        assert_class(garch, x=self.y, h=12, skip_insample=False, level=[90, 80])

    def test_garch_conformal_prediction(self):
        garch = GARCH(2, 2)
        fcst_garch = garch.forecast(ap, 12)

        garch_c = GARCH(
            2, 2, prediction_intervals=ConformalIntervals(h=13, n_windows=2)
        )
        assert_class(garch_c, x=ap, h=13, level=[90, 80], test_forward=False)
        fcst_garch_c = garch_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_garch_c["mean"][:12], fcst_garch["mean"])


# h = 100
# fcst = garch.forecast(y, h=h, level=[80, 95], fitted=True)
# fig, ax = plt.subplots(1, 1, figsize=(20, 7))
# # plt.plot(np.arange(0, len(y)), y)
# plt.plot(np.arange(len(y), len(y) + h), fcst["mean"], label="mean")
# plt.plot(np.arange(len(y), len(y) + h), fcst["sigma2"], color="c", label="sigma2")
# plt.plot(np.arange(len(y), len(y) + h), fcst["lo-95"], color="r", label="lo-95")
# plt.plot(np.arange(len(y), len(y) + h), fcst["hi-95"], color="r", label="hi-95")
# plt.plot(np.arange(len(y), len(y) + h), fcst["lo-80"], color="g", label="lo-80")
# plt.plot(np.arange(len(y), len(y) + h), fcst["hi-80"], color="g", label="hi-80")
# plt.legend()
# fig, ax = plt.subplots(1, 1, figsize=(20, 7))
# plt.plot(np.arange(0, len(y)), y)
# plt.plot(np.arange(0, len(y)), fcst["fitted"], label="fitted")
# plt.plot(np.arange(0, len(y)), fcst["fitted-lo-95"], color="r", label="fitted-lo-95")
# plt.plot(np.arange(0, len(y)), fcst["fitted-hi-95"], color="r", label="fitted-hi-95")
# plt.plot(np.arange(0, len(y)), fcst["fitted-lo-80"], color="g", label="fitted-lo-80")
# plt.plot(np.arange(0, len(y)), fcst["fitted-hi-80"], color="g", label="fitted-hi-80")
# plt.xlim(len(y) - 50, len(y))
# plt.legend()


class TestARCH:
    @classmethod
    def setup_class(cls):
        # Generate GARCH(2,2) data for testing
        n = 1000
        w = 0.5
        alpha = np.array([0.1, 0.2])
        beta = np.array([0.4, 0.2])
        cls.y = generate_garch_data(n, w, alpha, beta)

    def test_arch_basic(self):
        arch = ARCH(1)
        assert_class(arch, x=self.y, h=12, skip_insample=False, level=[90, 80])

    def test_arch_conformal_prediction(self):
        arch = ARCH(1)
        fcst_arch = arch.forecast(ap, 12)

        arch_c = ARCH(1, prediction_intervals=ConformalIntervals(h=13, n_windows=2))
        assert_class(arch_c, x=ap, h=13, level=[90, 80], test_forward=False)
        fcst_arch_c = arch_c.forecast(ap, 13, None, None, (80, 95), True)
        np.testing.assert_array_equal(fcst_arch_c["mean"][:12], fcst_arch["mean"])

    def test_arch_equals_garch_p1_q0(self):
        garch = GARCH(p=1, q=0)
        fcast_garch = garch.forecast(self.y, h=12, level=[90, 80], fitted=True)

        arch = ARCH(p=1)
        fcast_arch = arch.forecast(self.y, h=12, level=[90, 80], fitted=True)

        np.testing.assert_equal(fcast_garch, fcast_arch)


class TestSklearnModel:
    @classmethod
    def setup_class(cls):
        cls.h = 12
        cls.X = np.arange(ap.size).reshape(-1, 1)
        cls.X_future = ap.size + np.arange(cls.h).reshape(-1, 1)

    def test_sklearn_model_with_conformal_intervals(self):
        from sklearn.linear_model import Ridge

        skm = SklearnModel(Ridge(), prediction_intervals=ConformalIntervals(h=self.h))
        assert_class(
            skm,
            x=ap,
            X=self.X,
            X_future=self.X_future,
            h=self.h,
            skip_insample=False,
            level=[80, 95],
            test_forward=True,
        )

    def test_alias_arg(self):
        from sklearn.linear_model import Ridge

        assert repr(SklearnModel(Ridge())) == "Ridge"
        assert repr(SklearnModel(Ridge(), alias="my_ridge")) == "my_ridge"


class TestConstantModel:
    def test_constant_model_basic(self):
        constant_model = ConstantModel(constant=1)
        assert_class(constant_model, x=ap, h=12, level=[90, 80])
        constant_model.forecast(ap, 12, level=[90, 80])
        constant_model.forward(ap, 12, level=[90, 80])

    def test_alias_arg(self):
        assert repr(ConstantModel(1)) == "ConstantModel"
        assert (
            repr(ConstantModel(1, alias="ConstantModel_custom"))
            == "ConstantModel_custom"
        )


class TestZeroModel:
    def test_zero_model_basic(self):
        # Using the actual ZeroModel from statsforecast.models, not the test helper
        from statsforecast.models import ZeroModel as ActualZeroModel

        zero_model = ActualZeroModel()
        zero_model.fit(ap)
        result = zero_model.predict(h=12)
        assert len(result["mean"]) == 12
        assert all(result["mean"] == 0)  # ZeroModel should return all zeros

        # Test forecast method
        forecast_result = zero_model.forecast(ap, 12)
        assert len(forecast_result["mean"]) == 12
        assert all(forecast_result["mean"] == 0)

        # Test forward method
        forward_result = zero_model.forward(y=ap, h=12)
        assert len(forward_result["mean"]) == 12
        assert all(forward_result["mean"] == 0)

    def test_alias_arg(self):
        from statsforecast.models import ZeroModel as ActualZeroModel

        assert repr(ActualZeroModel()) == "ZeroModel"
        assert repr(ActualZeroModel(alias="ZeroModel_custom")) == "ZeroModel_custom"


class TestNaNModel:
    def test_nan_model_basic(self):
        nanmodel = NaNModel()
        nanmodel.forecast(ap, 12, level=[90, 80])

    def test_alias_arg(self):
        assert repr(NaNModel()) == "NaNModel"
        assert repr(NaNModel(alias="NaN_custom")) == "NaN_custom"
