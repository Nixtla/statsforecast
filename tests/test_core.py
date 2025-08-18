import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pltest
import pytest
from statsforecast.core import (
    ConformalIntervals,
    GroupedArray,
    StatsForecast,
    _get_n_jobs,
    _StatsForecast,
)
from statsforecast.models import (
    _TS,
    ADIDA,
    IMAPA,
    TSB,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    DynamicOptimizedTheta,
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    SeasonalExponentialSmoothing,
    SeasonalNaive,
    SeasonalWindowAverage,
    SimpleExponentialSmoothing,
    WindowAverage,
)
from statsforecast.utils import AirPassengers as ap
from statsforecast.utils import AirPassengersDF as ap_df
from statsforecast.utils import generate_series

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("always", category=UserWarning)
# logger.setLevel(logging.ERROR)


# sum ahead just returns the last value
# added with h future values
class SumAhead:
    def __init__(self):
        pass

    def fit(self, y, X):
        self.last_value = y[-1]
        self.fitted_values = np.full(y.size, np.nan, dtype=y.dtype)
        self.fitted_values[1:] = y[:1]
        return self

    def predict(self, h, X=None, level=None):
        mean = self.last_value + np.arange(1, h + 1)
        res = {"mean": mean}
        if level is not None:
            for lv in level:
                res[f"lo-{lv}"] = mean - 1.0
                res[f"hi-{lv}"] = mean + 1.0
        return res

    def __repr__(self):
        return "SumAhead"

    def forecast(self, y, h, X=None, X_future=None, fitted=False, level=None):
        mean = y[-1] + np.arange(1, h + 1)
        res = {"mean": mean}
        if fitted:
            fitted_values = np.full(y.size, np.nan, dtype=y.dtype)
            fitted_values[1:] = y[1:]
            res["fitted"] = fitted_values
        if level is not None:
            for lv in level:
                res[f"lo-{lv}"] = mean - 1.0
                res[f"hi-{lv}"] = mean + 1.0
        return res

    def forward(self, y, h, X=None, X_future=None, fitted=False, level=None):
        # fix self.last_value for test purposes
        mean = self.last_value + np.arange(1, h + 1)
        res = {"mean": mean}
        if fitted:
            fitted_values = np.full(y.size, np.nan, dtype=mean.dtype)
            fitted_values[1:] = y[1:]
            res["fitted"] = fitted_values
        if level is not None:
            for lv in level:
                res[f"lo-{lv}"] = mean - 1.0
                res[f"hi-{lv}"] = mean + 1.0
        return res

    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b


@pytest.fixture
def grouped_array_data():
    # data used for tests
    data = np.arange(12).reshape(-1, 1)
    indptr = np.array([0, 4, 8, 12])
    return GroupedArray(data, indptr), data, indptr


class NullModel(_TS):
    def __init__(self):
        pass

    def forecast(self):
        pass

    def __repr__(self):
        return "NullModel"


class FailedFit:
    def __init__(self):
        pass

    def forecast(self):
        pass

    def fit(self, y, X):
        raise Exception("Failed fit")

    def __repr__(self):
        return "FailedFit"


class TestGroupedArray:
    def test_groupedArray_length(self, grouped_array_data):
        # test we can recover the
        # number of series
        ga, data, indptr = grouped_array_data
        assert len(ga) == 3

    def test_splits(self, grouped_array_data):
        # test splits of data
        ga, data, indptr = grouped_array_data
        splits = ga.split(2)
        assert splits[0] == GroupedArray(data[:8], indptr[:3])
        assert splits[1] == GroupedArray(data[8:], np.array([0, 4]))

    def test_fit_and_predict_pipeline(self, grouped_array_data):
        ga, data, indptr = grouped_array_data

        # fitting models for each ts
        models = [Naive(), Naive()]
        fm = ga.fit(models)
        assert fm.shape == (3, 2)
        assert len(ga.split_fm(fm, 2)) == 2

        # test forecasts
        exp_fcsts = np.vstack([2 * [data[i]] for i in indptr[1:] - 1])
        fcsts, cols = ga.predict(fm=fm, h=2)
        np.testing.assert_equal(
            fcsts,
            np.hstack([exp_fcsts, exp_fcsts]),
        )

        # test fit and predict pipeline
        fm_fp, fcsts_fp, cols_fp = ga.fit_predict(models=models, h=2)
        assert fm_fp.shape == (3, 2)
        np.testing.assert_equal(fcsts_fp, fcsts)
        np.testing.assert_equal(cols_fp, cols)

        # test levels
        fm_lv, fcsts_lv, cols_lv = ga.fit_predict(models=models, h=2, level=(50, 90))
        assert fcsts_lv.shape == (2 * len(ga), 10)

        # test forecast
        fcst_f = ga.forecast(models=models, h=2, fitted=True)
        np.testing.assert_equal(fcst_f["forecasts"], fcsts_fp)
        np.testing.assert_equal(fcst_f["cols"], cols_fp)

        # test fallback model
        fcst_f = ga.forecast(
            models=[NullModel(), NullModel()], fallback_model=Naive(), h=2, fitted=True
        )
        np.testing.assert_equal(fcst_f["forecasts"], fcsts_fp)
        np.testing.assert_equal(fcst_f["cols"], ["NullModel", "NullModel"])
        with pytest.raises(Exception):
            ga.forecast(models=[NullModel()])

    def test_levels(self, grouped_array_data):
        ga, data, indptr = grouped_array_data

        # test levels
        lv = (50, 60)
        h = 2
        # test for forecasts
        fcsts_lv = ga.forecast(models=[SumAhead()], h=h, fitted=True, level=lv)
        assert fcsts_lv["forecasts"].shape == (len(ga) * h, 1 + 2 * len(lv))
        assert fcsts_lv["cols"] == [
            "SumAhead",
            "SumAhead-lo-50",
            "SumAhead-hi-50",
            "SumAhead-lo-60",
            "SumAhead-hi-60",
        ]
        # fit and predict pipeline
        fm_lv_fp, fcsts_lv_fp, cols_lv_fp = ga.fit_predict(
            models=[SumAhead()], h=h, level=lv
        )
        np.testing.assert_equal(fcsts_lv["forecasts"], fcsts_lv_fp)
        assert fcsts_lv["cols"] == cols_lv_fp

    def test_cross_validation(self):
        # tests for cross validation
        data = np.hstack([np.arange(10), np.arange(100, 200), np.arange(20, 40)])
        indptr = np.array([0, 10, 110, 130])
        ga = GroupedArray(data, indptr)

        res_cv = ga.cross_validation(models=[SumAhead()], h=2, test_size=5, fitted=True)
        fcsts_cv = res_cv["forecasts"]
        cols_cv = res_cv["cols"]
        np.testing.assert_equal(
            fcsts_cv[:, cols_cv.index("y")], fcsts_cv[:, cols_cv.index("SumAhead")]
        )

        # levels
        res_cv_lv = ga.cross_validation(
            models=[SumAhead(), Naive()], h=2, test_size=5, level=(50, 60)
        )
        actual_step_size = np.unique(
            np.diff(fcsts_cv[:, cols_cv.index("SumAhead")].reshape((3, -1, 2)), axis=1)
        )
        assert actual_step_size == 1

    def test_cross_validation_advanced(self):
        # tests for cross validation with different horizons, test sizes, and step sizes
        data = np.hstack([np.arange(10), np.arange(100, 200), np.arange(20, 40)])
        indptr = np.array([0, 10, 110, 130])
        ga = GroupedArray(data, indptr)

        horizons = [1, 2, 3, 2]
        test_sizes = [3, 4, 6, 6]
        step_sizes = [2, 2, 3, 4]
        for h, test_size, step_size in zip(horizons, test_sizes, step_sizes):
            res_cv = ga.cross_validation(
                models=[SumAhead()],
                h=h,
                test_size=test_size,
                step_size=step_size,
                fitted=True,
            )
            fcsts_cv = res_cv["forecasts"]
            cols_cv = res_cv["cols"]
            assert np.array_equal(
                fcsts_cv[:, cols_cv.index("y")], fcsts_cv[:, cols_cv.index("SumAhead")]
            )
            fcsts_cv = fcsts_cv[:, cols_cv.index("SumAhead")].reshape((3, -1, h))
            actual_step_size = np.unique(np.diff(fcsts_cv, axis=1))
            assert actual_step_size == step_size
            actual_n_windows = res_cv["forecasts"].shape[1]
            assert actual_n_windows == int((test_size - h) / step_size) + 1

        def fail_cv(h, test_size, step_size):
            return ga.cross_validation(
                models=[SumAhead()], h=h, test_size=test_size, step_size=step_size
            )

        with pytest.raises(Exception, match="module"):
            fail_cv(h=2, test_size=5, step_size=2)

        # test fallback model cross validation
        fcst_cv_f = ga.cross_validation(
            models=[NullModel(), NullModel()],
            fallback_model=Naive(),
            h=2,
            test_size=5,
            fitted=True,
        )
        fcst_cv_naive = ga.cross_validation(
            models=[Naive(), Naive()], h=2, test_size=5, fitted=True
        )
        assert np.array_equal(fcst_cv_f["forecasts"], fcst_cv_naive["forecasts"])
        np.testing.assert_array_equal(
            fcst_cv_f["fitted"]["values"], fcst_cv_naive["fitted"]["values"]
        )

        # test fallback model under failed fit for cross validation
        fcst_cv_f = ga.cross_validation(
            models=[FailedFit()],
            fallback_model=Naive(),
            h=2,
            test_size=5,
            refit=False,
            fitted=True,
        )
        fcst_cv_naive = ga.cross_validation(
            models=[Naive()],
            h=2,
            test_size=5,
            refit=False,
            fitted=True,
        )
        assert np.array_equal(fcst_cv_f["forecasts"], fcst_cv_naive["forecasts"])
        np.testing.assert_array_equal(
            fcst_cv_f["fitted"]["values"], fcst_cv_naive["fitted"]["values"]
        )

        # test cross validation without refit
        cv_starts = np.array([0, 8, 16])
        res_cv_wo_refit = ga.cross_validation(
            models=[SumAhead()], h=2, test_size=5, refit=False, level=(50, 60)
        )
        res_cv_refit = ga.cross_validation(
            models=[SumAhead()], h=2, test_size=5, refit=True, level=(50, 60)
        )

        # should fail when comparing different refit results
        with pytest.raises((AssertionError, ValueError)):
            assert np.array_equal(
                res_cv_wo_refit["forecasts"], res_cv_refit["forecasts"]
            )

        # test first forecasts are equal
        assert np.array_equal(
            res_cv_wo_refit["forecasts"][cv_starts],
            res_cv_refit["forecasts"][cv_starts],
        )

        # for refit=2 the first two windows should be the same
        res_cv_refit2 = ga.cross_validation(
            models=[SumAhead()], h=2, test_size=5, refit=2
        )
        assert np.array_equal(
            res_cv_refit2["forecasts"][np.hstack([cv_starts + 0, cv_starts + 1]), 1],
            res_cv_refit2["forecasts"][np.hstack([cv_starts + 2, cv_starts + 3]), 1],
        )
        # and the second two windows should be the same
        assert np.array_equal(
            res_cv_refit2["forecasts"][np.hstack([cv_starts + 4, cv_starts + 5]), 1],
            res_cv_refit2["forecasts"][np.hstack([cv_starts + 6, cv_starts + 7]), 1],
        )
        # but different between them
        with pytest.raises((AssertionError, ValueError)):
            assert np.array_equal(
                res_cv_refit2["forecasts"][
                    np.hstack([cv_starts + 0, cv_starts + 1]), 1
                ],
                res_cv_refit2["forecasts"][
                    np.hstack([cv_starts + 4, cv_starts + 5]), 1
                ],
            )

    def test_autoces_cross_validation_and_n_jobs(self):
        # tests for AutoCES cross validation with refit behavior
        data = np.hstack([np.arange(10), np.arange(100, 200), np.arange(20, 40)])
        indptr = np.array([0, 10, 110, 130])
        ga = GroupedArray(data, indptr)

        res_cv_wo_refit = ga.cross_validation(
            models=[AutoCES()], h=2, test_size=5, refit=False, level=(50, 60)
        )
        res_cv_refit = ga.cross_validation(
            models=[AutoCES()], h=2, test_size=5, refit=True, level=(50, 60)
        )

        # should fail when comparing different refit results
        with pytest.raises((AssertionError, ValueError)):
            assert np.array_equal(
                res_cv_wo_refit["forecasts"], res_cv_refit["forecasts"]
            )

        # test first forecasts are equal
        assert np.array_equal(
            res_cv_wo_refit["forecasts"][[0, 8, 16]],
            res_cv_refit["forecasts"][[0, 8, 16]],
        )

        # tests for more series than resources
        assert _get_n_jobs(100, -1) == os.cpu_count()
        assert _get_n_jobs(100, None) == os.cpu_count()
        assert _get_n_jobs(100, 2) == 2

        # tests for less series than resources
        assert _get_n_jobs(1, -1) == 1
        assert _get_n_jobs(1, None) == 1
        assert _get_n_jobs(2, 10) == 2


# # StatsForecast's class usage example


@pytest.fixture
def panel_df():
    # Generate synthetic panel DataFrame for example
    panel_df = generate_series(n_series=9, equal_ends=False, engine="pandas")
    panel_df.groupby("unique_id").tail(4)

    panel_with_exog = panel_df[panel_df["unique_id"] == 0].copy()

    return panel_df, panel_with_exog


def assert_raises_with_message(func, expected_msg, *args, **kwargs):
    with pytest.raises((AssertionError, ValueError, Exception)) as exc_info:
        func(*args, **kwargs)
    assert expected_msg in str(exc_info.value)


class TestModels:
    def test_autoarima(self, panel_df):
        panel_df, panel_with_exog = panel_df

        # if we train with a model that uses exog we must provide them through X_df
        # otherwise an error will be raised indicating this
        panel_with_exog["month"] = panel_df["ds"].dt.month
        sf = StatsForecast(
            models=[AutoARIMA(season_length=12)],
            freq="M",
        )
        sf.fit(panel_with_exog)
        expected_msg = (
            "['month'] for the forecasting step. Please provide them through `X_df`"
        )

        assert_raises_with_message(sf.predict, expected_msg, h=12)
        assert_raises_with_message(sf.forecast, expected_msg, df=panel_with_exog, h=12)
        assert_raises_with_message(
            sf.fit_predict, expected_msg, df=panel_with_exog, h=12
        )

    def test_SeasonalNaive(self, panel_df):
        panel_df, panel_with_exog = panel_df

        # if the models don't use exog then it continues
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=10), Naive()],
            freq="M",
        )
        sf.fit(panel_with_exog)
        _ = sf.predict(h=12)
        # checks for sizes with prediction intervals
        fcst = StatsForecast(models=[Naive()], freq="D")
        intervals = ConformalIntervals(n_windows=4, h=10)
        # intervals require 41 samples, with 30 we can use 2 windows, should warn
        with warnings.catch_warnings(record=True) as issued_warnings:
            fcst.fit(df=panel_df.head(30), prediction_intervals=intervals)
        assert "will use less windows" in str(issued_warnings[0].message)
        assert fcst.fitted_[0, 0]._cs.shape[0] == 2
        # if we have less than 21 samples (2 windows, h = 10 + 1 for training) it should fail

        msg = "Please remove them or adjust the horizon"
        assert_raises_with_message(
            fcst.fit, msg, df=panel_df.head(20), prediction_intervals=intervals
        )

        # for CV it should consider the test size (20 for CV, 21 for intervals)
        msg = "Minimum samples for computing prediction intervals are 41"
        assert_raises_with_message(
            fcst.cross_validation,
            msg,
            df=panel_df.head(40),
            n_windows=2,
            step_size=10,
            h=10,
            prediction_intervals=intervals,
            level=[80],
        )

        # integer refit or refit=False raises errors for unsupported models
        fcst = StatsForecast(models=[Naive(), RandomWalkWithDrift()], freq="D")
        assert_raises_with_message(
            fcst.cross_validation,
            "implement the forward method: [RWD]",
            df=panel_df,
            h=8,
            n_windows=4,
            refit=2,
        )
        assert_raises_with_message(
            fcst.cross_validation,
            "implement the forward method: [RWD]",
            df=panel_df,
            h=8,
            n_windows=4,
            refit=False,
        )
        fcst = StatsForecast(
            models=[Naive()], freq="D", fallback_model=RandomWalkWithDrift()
        )
        assert_raises_with_message(
            fcst.cross_validation,
            "a fallback model that implements the forward method.",
            df=panel_df,
            h=8,
            n_windows=4,
            refit=2,
        )
        assert_raises_with_message(
            fcst.cross_validation,
            "a fallback model that implements the forward method.",
            df=panel_df,
            h=8,
            n_windows=4,
            refit=False,
        )

    def test_non_standard_colnames(self, panel_df):
        panel_df, panel_with_exog = panel_df

        # non standard colnames
        fcst = StatsForecast(models=[Naive()], freq="D")
        fcst.fit(df=panel_df)
        std_preds = fcst.predict(h=1)
        std_preds2 = fcst.fit_predict(df=panel_df, h=1)
        std_fcst = fcst.forecast(df=panel_df, h=1, fitted=True)
        std_fitted = fcst.forecast_fitted_values()
        std_cv = fcst.cross_validation(df=panel_df, h=1, fitted=True)
        std_fitted_cv = fcst.cross_validation_fitted_values()

        renamer = {"unique_id": "uid", "ds": "time", "y": "target"}
        kwargs = dict(id_col="uid", time_col="time", target_col="target")
        inverse_renamer = {v: k for k, v in renamer.items()}
        non_std_df = panel_df.rename(columns=renamer)

        def assert_equal_results(df1, df2):
            pd.testing.assert_frame_equal(
                df1,
                df2.rename(columns=inverse_renamer),
            )

        fcst.fit(df=non_std_df, **kwargs)
        non_std_preds = fcst.predict(h=1)
        non_std_preds2 = fcst.fit_predict(df=non_std_df, h=1, **kwargs)
        non_std_fcst = fcst.forecast(df=non_std_df, h=1, fitted=True, **kwargs)
        non_std_fitted = fcst.forecast_fitted_values()
        non_std_cv = fcst.cross_validation(df=non_std_df, h=1, fitted=True, **kwargs)
        non_std_fitted_cv = fcst.cross_validation_fitted_values()

        assert_equal_results(std_preds, non_std_preds)
        assert_equal_results(std_preds2, non_std_preds2)
        assert_equal_results(std_fcst, non_std_fcst)
        assert_equal_results(std_fitted, non_std_fitted)
        assert_equal_results(std_cv, non_std_cv)
        assert_equal_results(std_fitted_cv, non_std_fitted_cv)

    def test_statsforecast_functionality(self, panel_df):
        panel_df, panel_with_exog = panel_df

        # Declare list of instantiated StatsForecast estimators to be fitted
        # You can try other estimator's hyperparameters
        # You can try other methods from the `models.StatsForecast` collection
        # Check them here: https://nixtlaverse.nixtla.io/statsforecast/models
        models = [
            AutoARIMA(),
            Naive(),
            AutoETS(),
            AutoARIMA(allowmean=True, alias="MeanAutoARIMA"),
        ]

        # Instantiate StatsForecast class
        fcst = StatsForecast(models=models, freq="D", n_jobs=1, verbose=True)

        # Efficiently predict
        fcsts_df = fcst.forecast(df=panel_df, h=4, fitted=True)
        fcsts_df.groupby("unique_id").tail(4)
        # Testing save and load

        with tempfile.TemporaryDirectory() as td:
            f_path = Path(td) / "sf_test.pickle"

            test_df = generate_series(n_series=9, equal_ends=False, engine="polars")
            test_frcs = StatsForecast(models=models, freq="1d", n_jobs=1, verbose=True)

            origin_df = test_frcs.forecast(df=test_df, h=4, fitted=True)

            test_frcs.save(f_path)

            sf_test = StatsForecast.load(f_path)
            load_df = sf_test.forecast(df=test_df, h=4, fitted=True)

            pl.testing.assert_frame_equal(origin_df, load_df)
        # test custom names
        assert fcsts_df.columns[-1] == "MeanAutoARIMA"
        # test no duplicate names
        assert_raises_with_message(
            StatsForecast, "", models=[Naive(), Naive()], freq="D"
        )
        StatsForecast(models=[Naive(), Naive(alias="Naive2")], freq="D")
        fig = StatsForecast.plot(panel_df, max_insample_length=10)
        fig
        assert_raises_with_message(
            StatsForecast.plot,
            "Please use a list",
            df=panel_df,
            level=90,
        )

    def test_model_prediction_interval_overrides(self):
        # test model prediction_interval overrides
        models = [
            SimpleExponentialSmoothing(
                alpha=0.1, prediction_intervals=ConformalIntervals(h=24, n_windows=2)
            )
        ]
        fcst = StatsForecast(models=models, freq="D", n_jobs=1)
        fcst._set_prediction_intervals(None)
        assert models[0].prediction_intervals is not None


# fcst = StatsForecast(
#     models=[AutoARIMA(season_length=7)], freq="D", n_jobs=1, verbose=True
# )
# fcsts_df = fcst.forecast(df=panel_df, h=4, fitted=True, level=[90])
# fcsts_df.groupby("unique_id").tail(4)
# fitted_vals = fcst.forecast_fitted_values()
# fcst.plot(panel_df, fitted_vals.drop(columns="y"), level=[90])
# show_doc(_StatsForecast.fit, title_level=2, name="StatsForecast.fit")
# show_doc(_StatsForecast.predict, title_level=2, name="SatstForecast.predict")
# show_doc(_StatsForecast.fit_predict, title_level=2, name="StatsForecast.fit_predict")
# show_doc(_StatsForecast.forecast, title_level=2, name="StatsForecast.forecast")
# # StatsForecast.forecast method usage example

# from statsforecast.core import StatsForecast

# Instantiate StatsForecast class
# fcst = StatsForecast(models=[AutoARIMA(), Naive()], freq="D", n_jobs=1)

# Efficiently predict without storing memory
# fcsts_df = fcst.forecast(df=ap_df, h=4, fitted=True)
# fcsts_df.groupby("unique_id").tail(4)


@pytest.fixture
def series():
    series = generate_series(100, n_static_features=2, equal_ends=False)
    return series


def models_to_test():
    models = [
        ADIDA(),
        CrostonClassic(),
        CrostonOptimized(),
        CrostonSBA(),
        HistoricAverage(),
        IMAPA(),
        Naive(),
        RandomWalkWithDrift(),
        SeasonalExponentialSmoothing(season_length=7, alpha=0.1),
        SeasonalNaive(season_length=7),
        SeasonalWindowAverage(season_length=7, window_size=4),
        SimpleExponentialSmoothing(alpha=0.1),
        TSB(alpha_d=0.1, alpha_p=0.3),
        WindowAverage(window_size=4),
    ]
    return models


@pytest.mark.parametrize("models", models_to_test())
def test_series_without_ds_as_datetime(series, models):
    fcst = StatsForecast(models=[models], freq="D", n_jobs=1, verbose=True)

    res = fcst.forecast(df=series, h=14)
    # test series without ds as datetime
    series_wo_dt = series.copy()
    series_wo_dt["ds"] = series_wo_dt["ds"].astype(str)
    fcst = StatsForecast(models=[models], freq="D")
    fcsts_wo_dt = fcst.forecast(df=series_wo_dt, h=14)
    pd.testing.assert_frame_equal(res, fcsts_wo_dt)
    np.testing.assert_array_equal(res["unique_id"].unique(), fcst.uids.values)
    last_dates = series.groupby("unique_id", observed=True)["ds"].max()
    np.testing.assert_array_equal(
        res.groupby("unique_id", observed=True)["ds"].min().values,
        last_dates + pd.offsets.Day(),
    )
    np.testing.assert_array_equal(
        res.groupby("unique_id", observed=True)["ds"].max().values,
        last_dates + 14 * pd.offsets.Day(),
    )


def test_for_monthly_data():
    # tests for monthly data

    monthly_series = generate_series(
        10_000, freq="M", min_length=10, max_length=20, equal_ends=True
    )

    fcst = StatsForecast(models=[Naive()], freq="M")
    monthly_res = fcst.forecast(df=monthly_series, h=4)

    # last_dates = monthly_series.groupby("unique_id")["ds"].max()
    np.testing.assert_array_equal(
        monthly_res.groupby("unique_id", observed=True)["ds"].min().values,
        pd.Series(fcst.last_dates) + pd.offsets.MonthEnd(),
    )
    np.testing.assert_array_equal(
        monthly_res.groupby("unique_id", observed=True)["ds"].max().values,
        pd.Series(fcst.last_dates) + 4 * pd.offsets.MonthEnd(),
    )


# # StatsForecast.forecast_fitted_values method usage example

# # from statsforecast.core import StatsForecast
# from statsforecast.models import Naive
# from statsforecast.utils import AirPassengersDF as panel_df

# # Instantiate StatsForecast class
# fcst = StatsForecast(models=[AutoARIMA()], freq="D", n_jobs=1)

# # Access insample predictions
# fcsts_df = fcst.forecast(df=panel_df, h=12, fitted=True, level=(90, 10))
# insample_fcsts_df = fcst.forecast_fitted_values()
# insample_fcsts_df.tail(4)


def assert_fcst_fitted(series, n_jobs=1, str_ds=False):
    # tests for fitted values
    if str_ds:
        series = series.copy()
        series["ds"] = series["ds"].astype(str)
    fitted_fcst = StatsForecast(
        models=[Naive()],
        freq="D",
        n_jobs=n_jobs,
    )
    fitted_res = fitted_fcst.forecast(df=series, h=14, fitted=True)
    fitted = fitted_fcst.forecast_fitted_values()
    if str_ds:
        np.testing.assert_array_equal(pd.to_datetime(series["ds"]), fitted["ds"])
    else:
        np.testing.assert_array_equal(series["ds"], fitted["ds"])
    np.testing.assert_array_equal(series["y"], fitted["y"])


@pytest.mark.parametrize("str_ds", [True, False])
def test_for_fitted_values(series, str_ds):
    assert_fcst_fitted(series, str_ds=str_ds)


def test_fcst_fallback_model(series, n_jobs=1):
    # tests for fallback model

    fitted_fcst = StatsForecast(
        models=[NullModel()], freq="D", n_jobs=n_jobs, fallback_model=Naive()
    )
    fitted_res = fitted_fcst.forecast(df=series, h=14, fitted=True)
    fitted = fitted_fcst.forecast_fitted_values()
    np.testing.assert_array_equal(series["ds"], fitted["ds"])
    np.testing.assert_array_equal(series["y"], fitted["y"])
    # test NullModel actualy fails
    fitted_fcst = StatsForecast(
        models=[NullModel()],
        freq="D",
        n_jobs=n_jobs,
    )
    assert_raises_with_message(fitted_fcst.forecast, "", df=series, h=14)


# show_doc(
#     _StatsForecast.cross_validation,
#     title_level=2,
#     name="StatsForecast.cross_validation",
# )
# # StatsForecast.crossvalidation method usage example

# # from statsforecast.core import StatsForecast
# from statsforecast.models import Naive
# from statsforecast.utils import AirPassengersDF as panel_df

# # Instantiate StatsForecast class
# fcst = StatsForecast(models=[Naive()], freq="D", n_jobs=1, verbose=True)

# # Access insample predictions
# rolled_fcsts_df = fcst.cross_validation(df=panel_df, h=14, n_windows=2)
# rolled_fcsts_df.head(4)


@pytest.fixture
def series_cv1():
    series_cv = pd.DataFrame(
        {
            "unique_id": np.array(10 * ["id_0"] + 100 * ["id_1"] + 20 * ["id_2"]),
            "ds": np.hstack(
                [
                    pd.date_range(end="2021-01-01", freq="D", periods=10),
                    pd.date_range(end="2022-01-01", freq="D", periods=100),
                    pd.date_range(end="2020-01-01", freq="D", periods=20),
                ]
            ),
            "y": np.hstack([np.arange(10.0), np.arange(100, 200), np.arange(20, 40)]),
        }
    )
    return series_cv


@pytest.fixture
def series_cv2():
    series_cv = pd.DataFrame(
        {
            "unique_id": np.hstack(
                [np.zeros(10), np.zeros(100) + 1, np.zeros(20) + 2]
            ).astype("int64"),
            "ds": np.hstack(
                [
                    pd.date_range(end="2022-01-01", freq="D", periods=10),
                    pd.date_range(end="2022-01-01", freq="D", periods=100),
                    pd.date_range(end="2022-01-01", freq="D", periods=20),
                ]
            ),
            "y": np.hstack([np.arange(10), np.arange(100, 200), np.arange(20, 40)]),
        }
    )
    return series_cv


def test_for_cross_validation(series_cv1):
    series_cv = series_cv1
    # test for cross_validation
    fcst = StatsForecast(
        models=[SumAhead(), Naive()],
        freq="D",
        verbose=True,
    )
    res_cv = fcst.cross_validation(
        df=series_cv, h=2, test_size=5, n_windows=None, level=(50, 60)
    )
    np.testing.assert_array_equal(0.0, np.mean(res_cv["y"] - res_cv["SumAhead"]))

    n_windows = (
        fcst.cross_validation(df=series_cv, h=2, n_windows=2)
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 2 * 2
    np.testing.assert_array_equal(0.0, np.mean(res_cv["y"] - res_cv["SumAhead"]))

    n_windows = (
        fcst.cross_validation(df=series_cv, h=3, n_windows=3, step_size=3, fitted=True)
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 3 * 3
    np.testing.assert_array_equal(0.0, np.mean(res_cv["y"] - res_cv["SumAhead"]))

    assert_raises_with_message(
        fcst.cross_validation,
        "The following series are too short for the cross validation settings: ['id_0']",
        df=series_cv,
        h=10,
    )
    assert_raises_with_message(
        fcst.cross_validation,
        "The following series are too short for the cross validation settings: ['id_0', 'id_2']",
        df=series_cv,
        h=20,
    )


def test_cross_validation_refit_false(series_cv1):
    series_cv = series_cv1

    # test cross validation refit=False with SumAhead
    fcst = StatsForecast(models=[SumAhead()], freq="D", verbose=True)
    res_cv = fcst.cross_validation(
        df=series_cv,
        h=2,
        test_size=5,
        n_windows=None,
        level=(50, 60),
        refit=True,
    )
    res_cv_wo_refit = fcst.cross_validation(
        df=series_cv,
        h=2,
        test_size=5,
        n_windows=None,
        level=(50, 60),
        refit=False,
    )

    # Results should be different
    with pytest.raises((AssertionError, ValueError)):
        pd.testing.assert_frame_equal(res_cv_wo_refit, res_cv)

    # But first predictions should be equal
    cols_wo_refit = res_cv_wo_refit.columns
    pd.testing.assert_frame_equal(
        res_cv_wo_refit.groupby("unique_id").head(1),
        res_cv[cols_wo_refit].groupby("unique_id").head(1),
    )

    # Test n_windows with refit=False
    n_windows = (
        fcst.cross_validation(
            df=series_cv,
            h=2,
            n_windows=2,
            refit=False,
        )
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 2 * 2

    n_windows = (
        fcst.cross_validation(
            df=series_cv,
            h=3,
            n_windows=3,
            step_size=3,
            fitted=True,
            refit=False,
        )
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 3 * 3


def test_cross_validation_refit_false_complex_models(series_cv2):
    series_cv = series_cv2
    # test cross validation refit=False with complex models
    fcst = StatsForecast(
        models=[
            DynamicOptimizedTheta(),
            AutoCES(),
            DynamicOptimizedTheta(season_length=7, alias="test"),
        ],
        freq="D",
        verbose=True,
    )
    res_cv_wo_refit = fcst.cross_validation(
        df=series_cv,
        h=2,
        test_size=5,
        n_windows=None,
        level=(50, 60),
        refit=False,
    )
    res_cv_w_refit = fcst.cross_validation(
        df=series_cv,
        h=2,
        test_size=5,
        n_windows=None,
        level=(50, 60),
        refit=True,
    )

    # Results should be different
    with pytest.raises((AssertionError, ValueError)):
        pd.testing.assert_frame_equal(res_cv_wo_refit, res_cv_w_refit)

    # But first predictions should be equal
    pd.testing.assert_frame_equal(
        res_cv_wo_refit.groupby("unique_id").head(1),
        res_cv_w_refit.groupby("unique_id").head(1),
    )


def test_cross_validation_string_dates(series_cv2):
    series_cv = series_cv2

    # test series without ds as datetime
    series_cv_wo_dt = series_cv.copy()
    series_cv_wo_dt["ds"] = series_cv_wo_dt["ds"].astype(str)
    fcst = StatsForecast(models=[SumAhead(), Naive()], freq="D", verbose=False)

    res_cv = fcst.cross_validation(
        df=series_cv,
        h=2,
        test_size=5,
        n_windows=None,
        level=(50, 60),
    )
    res_cv_wo_dt = fcst.cross_validation(
        df=series_cv_wo_dt,
        h=2,
        test_size=5,
        n_windows=None,
        level=(50, 60),
    )
    pd.testing.assert_frame_equal(res_cv, res_cv_wo_dt)


def test_cross_validation_equal_ends():
    # test for equal ends cross_validation
    series_cv = pd.DataFrame(
        {
            "unique_id": np.hstack(
                [np.zeros(10), np.zeros(100) + 1, np.zeros(20) + 2]
            ).astype("int64"),
            "ds": np.hstack(
                [
                    pd.date_range(end="2022-01-01", freq="D", periods=10),
                    pd.date_range(end="2022-01-01", freq="D", periods=100),
                    pd.date_range(end="2022-01-01", freq="D", periods=20),
                ]
            ),
            "y": np.hstack([np.arange(10), np.arange(100, 200), np.arange(20, 40)]),
        }
    )

    fcst = StatsForecast(
        models=[SumAhead()],
        freq="D",
    )
    res_cv = fcst.cross_validation(
        df=series_cv,
        h=2,
        test_size=5,
        n_windows=None,
        level=(50, 60),
        fitted=True,
    )
    assert 0.0 == np.mean(res_cv["y"] - res_cv["SumAhead"])

    n_windows = (
        fcst.cross_validation(
            df=series_cv,
            h=2,
            n_windows=2,
        )
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 2 * 2
    assert 0.0 == np.mean(res_cv["y"] - res_cv["SumAhead"])

    n_windows = (
        fcst.cross_validation(
            df=series_cv,
            h=3,
            n_windows=3,
            step_size=3,
        )
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 3 * 3
    assert 0.0 == np.mean(res_cv["y"] - res_cv["SumAhead"])


# show_doc(
#     _StatsForecast.cross_validation_fitted_values,
#     title_level=2,
#     name="StatsForecast.cross_validation_fitted_values",
# )
# # StatsForecast.cross_validation_fitted_values method usage example

# # from statsforecast.core import StatsForecast
# from statsforecast.models import Naive
# from statsforecast.utils import AirPassengersDF as panel_df


# # Instantiate StatsForecast class
def test_naive_model(series):
    fcst = StatsForecast(models=[Naive()], freq="D", n_jobs=1)
    fcst.fit(df=series)
    np.testing.assert_array_equal(fcst.predict(h=12), fcst.forecast(df=series, h=12))
    np.testing.assert_array_equal(
        fcst.fit_predict(df=series, h=12), fcst.forecast(df=series, h=12)
    )


# # Access insample predictions
# rolled_fcsts_df = fcst.cross_validation(df=panel_df, h=12, n_windows=2, fitted=True)
# insample_rolled_fcsts_df = fcst.cross_validation_fitted_values()
# insample_rolled_fcsts_df.tail(4)


# tests for fitted values cross_validation
def assert_cv_fitted(series_cv2, n_jobs=1, str_ds=False):
    series_cv = series_cv2
    if str_ds:
        series_cv = series_cv.copy()
        series_cv["ds"] = series_cv["ds"].astype(str)
    resids_fcst = StatsForecast(models=[SumAhead(), Naive()], freq="D", n_jobs=n_jobs)
    resids_res_cv = resids_fcst.cross_validation(
        df=series_cv, h=2, n_windows=4, fitted=True
    )
    resids_cv = resids_fcst.cross_validation_fitted_values()
    np.testing.assert_array_equal(
        resids_cv["cutoff"].unique(), resids_res_cv["cutoff"].unique()
    )
    if str_ds:
        series_cv["ds"] = pd.to_datetime(series_cv["ds"])
    for uid in resids_cv["unique_id"].unique():
        resids_uid = resids_cv[resids_cv["unique_id"].eq(uid)]
        for cutoff in resids_uid["cutoff"].unique():
            pd.testing.assert_frame_equal(
                resids_uid.query("cutoff == @cutoff")[
                    ["unique_id", "ds", "y"]
                ].reset_index(drop=True),
                series_cv.query("ds <= @cutoff & unique_id == @uid")[
                    ["unique_id", "ds", "y"]
                ].reset_index(drop=True),
                check_dtype=False,
            )


@pytest.mark.parametrize("str_ds", [True, False])
def test_cv_fitted(series_cv2, str_ds):
    assert_cv_fitted(series_cv2, str_ds=str_ds)


# tests for fallback model
def test_cv_fallback_model(series, n_jobs=1):
    fitted_fcst = StatsForecast(
        models=[NullModel()], freq="D", n_jobs=n_jobs, fallback_model=Naive()
    )
    fitted_res = fitted_fcst.cross_validation(df=series, h=2, n_windows=4, fitted=True)
    fitted = fitted_fcst.cross_validation_fitted_values()
    # test NullModel actualy fails
    fitted_fcst = StatsForecast(
        models=[NullModel()],
        freq="D",
        n_jobs=n_jobs,
    )
    (
        assert_raises_with_message(
            fitted_fcst.cross_validation,
            "got an unexpected keyword argument",
            df=series,
            h=12,
            n_windows=4,
        ),
    )


# test_cv_fallback_model()
# show_doc(_StatsForecast.plot, title_level=2, name="StatsForecast.plot")
# show_doc(StatsForecast.save, title_level=2, name="StatsForecast.save")
# show_doc(StatsForecast.load, title_level=2, name="StatsForecast.load")


def test_conformal_prediction(series):
    # test for conformal prediction
    uids = series.index.unique()[:10]
    series_subset = series.query("unique_id in @uids")[["unique_id", "ds", "y"]]
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=7)],
        freq="D",
        n_jobs=1,
    )
    sf = sf.fit(df=series_subset, prediction_intervals=ConformalIntervals(h=12))
    np.testing.assert_array_equal(
        sf.predict(h=12, level=[80, 90]),
        sf.fit_predict(
            df=series_subset,
            h=12,
            level=[80, 90],
            prediction_intervals=ConformalIntervals(h=12),
        ),
    )
    np.testing.assert_array_equal(
        sf.predict(h=12, level=[80, 90]),
        sf.forecast(
            df=series_subset,
            h=12,
            level=[80, 90],
            prediction_intervals=ConformalIntervals(h=12),
        ),
    )

    # test errors/warnings are raised when level is not specified
    intervals = ConformalIntervals(h=12)
    sf2 = StatsForecast(
        models=[ADIDA()],
        freq="D",
        n_jobs=1,
    )
    sf2.fit(df=series_subset, prediction_intervals=intervals)
    with pytest.warns():
        sf2.predict(h=12)

    # test_warns(lambda: )
    assert_raises_with_message(
        sf2.forecast, "", df=series_subset, h=12, prediction_intervals=intervals
    )
    assert_raises_with_message(
        sf2.fit_predict, "", df=series_subset, h=12, prediction_intervals=intervals
    )
    assert_raises_with_message(
        sf2.cross_validation, "", df=series_subset, h=12, prediction_intervals=intervals
    )

    # test conformal cross validation
    cv_conformal = sf.cross_validation(
        df=series_subset,
        h=12,
        n_windows=2,
        level=[80, 90],
        prediction_intervals=ConformalIntervals(h=12),
    )
    cv_no_conformal = sf.cross_validation(
        df=series_subset,
        h=12,
        n_windows=2,
        level=[80, 90],
    )
    np.testing.assert_array_equal(
        cv_conformal.columns,
        cv_no_conformal.columns,
    )
    np.testing.assert_array_equal(
        cv_conformal.filter(regex="ds|cutoff|y|AutoARIMA$"),
        cv_no_conformal.filter(regex="ds|cutoff|y|AutoARIMA$"),
    )


# fcst = StatsForecast(
#     models=[
#         ADIDA(),
#         SimpleExponentialSmoothing(0.1),
#         HistoricAverage(),
#         CrostonClassic(),
#     ],
#     freq="D",
#     n_jobs=1,
# )
# res = fcst.forecast(df=series, h=14)
# # | eval: false
def test_parallel_processing(series, series_cv2):
    series_cv = series_cv2
    # tests for parallel processing
    fcst = StatsForecast(
        models=[
            ADIDA(),
            SimpleExponentialSmoothing(0.1),
            HistoricAverage(),
            CrostonClassic(),
        ],
        freq="D",
        n_jobs=-1,
    )
    res = fcst.forecast(df=series, h=14)
    res_cv = fcst.cross_validation(df=series, h=3, test_size=10, n_windows=None)
    fcst = StatsForecast(
        models=[SumAhead()],
        freq="D",
    )
    res_cv = fcst.cross_validation(df=series_cv, h=2, test_size=5, n_windows=None)
    np.testing.assert_array_equal(0.0, np.mean(res_cv["y"] - res_cv["SumAhead"]))

    assert_fcst_fitted(series)
    assert_cv_fitted(series_cv, n_jobs=-1)
    assert_fcst_fitted(series, n_jobs=-1, str_ds=True)
    assert_cv_fitted(series_cv, n_jobs=-1, str_ds=True)
    # check n_windows argument
    n_windows = (
        fcst.cross_validation(df=series_cv, h=2, n_windows=2)
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 2 * 2
    np.testing.assert_array_equal(0.0, np.mean(res_cv["y"] - res_cv["SumAhead"]))

    # check step_size argument
    n_windows = (
        fcst.cross_validation(df=series_cv, h=3, n_windows=3, step_size=3)
        .groupby("unique_id")
        .size()
        .unique()
    )
    assert n_windows == 3 * 3
    np.testing.assert_array_equal(0.0, np.mean(res_cv["y"] - res_cv["SumAhead"]))


def test_integer_datestamp():
    int_ds_df = pd.DataFrame({"ds": np.arange(1, len(ap) + 1), "y": ap})
    int_ds_df.insert(0, "unique_id", "AirPassengers")

    fcst = StatsForecast(models=[HistoricAverage()], freq=1)
    horizon = 7
    forecast = fcst.forecast(df=int_ds_df, h=horizon)

    last_date = int_ds_df["ds"].max()
    np.testing.assert_array_equal(
        forecast["ds"].values, np.arange(last_date + 1, last_date + 1 + horizon)
    )


# int_ds_cv = fcst.cross_validation(df=int_ds_df, h=7, test_size=8, n_windows=None)
# int_ds_cv


# class LinearRegression(_TS):
#     def __init__(self):
#         pass

#     def fit(self, y, X):
#         self.coefs_, *_ = np.linalg.lstsq(X, y, rcond=None)
#         return self

#     def predict(self, h, X):
#         mean = X @ coefs
#         return mean

#     def __repr__(self):
#         return "LinearRegression()"

#     def forecast(self, y, h, X=None, X_future=None, fitted=False):
#         coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
#         return {"mean": X_future @ coefs}

#     def new(self):
#         b = type(self).__new__(type(self))
#         b.__dict__.update(self.__dict__)
#         return b


# series_xreg = series = generate_series(10_000, equal_ends=True)
# series_xreg["intercept"] = 1
# series_xreg["dayofweek"] = series_xreg["ds"].dt.dayofweek
# series_xreg = pd.get_dummies(series_xreg, columns=["dayofweek"], drop_first=True)
# series_xreg
# dates = sorted(series_xreg["ds"].unique())
# valid_start = dates[-14]
# train_mask = series_xreg["ds"] < valid_start
# series_train = series_xreg[train_mask]
# series_valid = series_xreg[~train_mask]
# X_valid = series_valid.drop(columns=["y"])
# fcst = StatsForecast(
#     models=[LinearRegression()],
#     freq="D",
# )
# xreg_res = fcst.forecast(df=series_train, h=14, X_df=X_valid)
# xreg_res["y"] = series_valid["y"].values
# xreg_res.drop(columns="unique_id").groupby("ds").mean().plot()
# xreg_res_cv = fcst.cross_validation(df=series_train, h=3, test_size=5, n_windows=None)


# the following cells contain tests for external regressors
class ReturnX(_TS):
    def __init__(self):
        pass

    def fit(self, y, X):
        return self

    def predict(self, h, X):
        mean = X
        return X

    def __repr__(self):
        return "ReturnX"

    def forecast(self, y, h, X=None, X_future=None, fitted=False):
        return {"mean": X_future.flatten()}

    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_x_vars(n_jobs):
    df = pd.DataFrame(
        {
            "unique_id": [0] * 10 + [1] * 10,
            "ds": np.hstack([np.arange(10), np.arange(10)]),
            "y": np.random.rand(20),
            "x": np.arange(20, dtype=np.float64),
        }
    )
    train_mask = df["ds"] < 6
    train_df = df[train_mask]
    test_df = df[~train_mask]
    fcst = StatsForecast(
        models=[ReturnX()],
        freq=1,
        n_jobs=n_jobs,
    )
    xreg = test_df.drop(columns="y")
    res = fcst.forecast(df=train_df, h=4, X_df=xreg)
    expected_res = xreg.rename(columns={"x": "ReturnX"})
    pd.testing.assert_frame_equal(
        res,
        expected_res.reset_index(drop=True),
        check_dtype=False,
    )


# test_x_vars(n_jobs=1)
# # | eval: false
# test_x_vars(n_jobs=2)
# ap_df = pd.DataFrame({"ds": np.arange(ap.size), "y": ap})
# ap_df["unique_id"] = 0
# sf = StatsForecast(
#     models=[SeasonalNaive(season_length=12), AutoARIMA(season_length=12)],
#     freq=1,
#     n_jobs=1,
# )
# ap_ci = sf.forecast(df=ap_df, h=12, level=(80, 95))
# fcst.plot(ap_df, ap_ci, level=[80], engine="matplotlib")
# from statsforecast.utils import ConformalIntervals

# sf = StatsForecast(
#     models=[
#         AutoARIMA(season_length=12),
#         AutoARIMA(
#             season_length=12,
#             prediction_intervals=ConformalIntervals(n_windows=2, h=12),
#             alias="ConformalAutoARIMA",
#         ),
#     ],
#     freq=1,
#     n_jobs=1,
# )
# ap_ci = sf.forecast(df=ap_df, h=12, level=(80, 95))
# fcst.plot(ap_df, ap_ci, level=[80], engine="plotly")
# sf = StatsForecast(
#     models=[
#         AutoARIMA(season_length=12),
#     ],
#     freq=1,
#     n_jobs=1,
# )
# ap_ci = sf.forecast(
#     df=ap_df,
#     h=12,
#     level=(50, 80, 95),
#     prediction_intervals=ConformalIntervals(h=12),
# )
# fcst.plot(ap_df, ap_ci, level=[80], engine="matplotlib")


@pytest.mark.parametrize("n_jobs", [1, 101])
def test_conf_intervals(n_jobs):
    ap_df = pd.DataFrame(
        {"unique_id": [0] * ap.size, "ds": np.arange(ap.size), "y": ap}
    )
    fcst = StatsForecast(
        models=[SeasonalNaive(season_length=12), AutoARIMA(season_length=12)],
        freq=1,
        n_jobs=n_jobs,
    )
    ap_ci = fcst.forecast(df=ap_df, h=12, level=(80, 95))
    ap_ci.drop(columns="unique_id").set_index("ds").plot(marker=".", figsize=(10, 6))


# test_conf_intervals(n_jobs=1)
# # | eval: false
# # test number of jobs greater than the available cores
# test_conf_intervals(n_jobs=101)
