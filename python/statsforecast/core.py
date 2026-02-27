__all__ = ["StatsForecast"]


import datetime as dt
import errno
import inspect
import logging
import os
import pickle
import re
import reprlib
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from fugue.execution.factory import (
    make_execution_engine,
    try_get_context_execution_engine,
)
from threadpoolctl import ThreadpoolController
from tqdm.auto import tqdm
from triad import conditional_dispatcher
from utilsforecast.compat import DataFrame, pl_DataFrame, pl_Series
from utilsforecast.grouped_array import GroupedArray as BaseGroupedArray
from utilsforecast.validation import ensure_time_dtype, validate_freq

from .distributions import _get_distribution
from .utils import ConformalIntervals, _ensure_float

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger(__name__)
_controller = ThreadpoolController()


class GroupedArray(BaseGroupedArray):
    def __init__(self, data: np.ndarray, indptr: np.ndarray):
        self.data = _ensure_float(data)
        self.indptr = indptr
        self.n_groups = len(indptr) - 1

    def __eq__(self, other):
        if not hasattr(other, "data") or not hasattr(other, "indptr"):
            return False
        return np.allclose(self.data, other.data) and np.array_equal(
            self.indptr, other.indptr
        )

    def fit(self, models, fallback_model=None):
        fm = np.full((self.n_groups, len(models)), np.nan, dtype=object)
        for i, grp in enumerate(self):
            y = grp[:, 0] if grp.ndim == 2 else grp
            X = grp[:, 1:] if (grp.ndim == 2 and grp.shape[1] > 1) else None
            for i_model, model in enumerate(models):
                try:
                    new_model = model.new()
                    fm[i, i_model] = new_model.fit(y=y, X=X)
                except Exception as error:
                    if fallback_model is not None:
                        new_fallback_model = fallback_model.new()
                        new_fallback_model.alias = model.alias
                        fm[i, i_model] = new_fallback_model.fit(y=y, X=X)
                    else:
                        raise error
        return fm

    def _get_cols(self, models, attr, h, X, level=tuple()):
        n_models = len(models)
        cuts = np.full(n_models + 1, fill_value=0, dtype=np.int32)
        has_level_models = np.full(n_models, fill_value=False, dtype=bool)
        cuts[0] = 0
        for i_model, model in enumerate(models):
            len_cols = 1  # mean
            has_level = (
                "level" in inspect.signature(getattr(model, attr)).parameters
                and len(level) > 0
            )
            has_level_models[i_model] = has_level
            if has_level:
                len_cols += 2 * len(level)  # levels
            cuts[i_model + 1] = len_cols + cuts[i_model]
        return cuts, has_level_models

    def _output_fcst(self, models, attr, h, X, level=tuple()):
        # returns empty output according to method
        cuts, has_level_models = self._get_cols(
            models=models, attr=attr, h=h, X=X, level=level
        )
        out = np.full(
            (self.n_groups * h, cuts[-1]), fill_value=np.nan, dtype=self.data.dtype
        )
        return out, cuts, has_level_models

    def predict(self, fm, h, X=None, level=tuple()):
        # fm stands for fitted_models
        # and fm should have fitted_model
        fcsts, cuts, has_level_models = self._output_fcst(
            models=fm[0], attr="predict", h=h, X=X, level=level
        )
        matches = ["mean", "lo", "hi"]
        cols = []
        for i_model in range(fm.shape[1]):
            has_level = has_level_models[i_model]
            kwargs = {}
            if has_level:
                kwargs["level"] = level
            for i, _ in enumerate(self):
                if X is not None:
                    X_ = X[i]
                else:
                    X_ = None
                res_i = fm[i, i_model].predict(h=h, X=X_, **kwargs)
                cols_m = [
                    key
                    for key in res_i.keys()
                    if any(key.startswith(m) for m in matches)
                ]
                fcsts_i = np.vstack([res_i[key] for key in cols_m]).T
                model_name = repr(fm[i, i_model])
                cols_m = [
                    f"{model_name}" if col == "mean" else f"{model_name}-{col}"
                    for col in cols_m
                ]
                if fcsts_i.ndim == 1:
                    fcsts_i = fcsts_i[:, None]
                fcsts[i * h : (i + 1) * h, cuts[i_model] : cuts[i_model + 1]] = fcsts_i
            cols += cols_m
        return fcsts, cols

    def fit_predict(self, models, h, X=None, level=tuple()):
        # fitted models
        fm = self.fit(models=models)
        # forecasts
        fcsts, cols = self.predict(fm=fm, h=h, X=X, level=level)
        return fm, fcsts, cols

    def forecast(
        self,
        models,
        h,
        fallback_model=None,
        fitted=False,
        X=None,
        level=tuple(),
        verbose=False,
        target_col="y",
    ):
        fcsts, cuts, has_level_models = self._output_fcst(
            models=models, attr="forecast", h=h, X=X, level=level
        )
        matches = ["mean", "lo", "hi"]
        matches_fitted = ["fitted", "fitted-lo", "fitted-hi"]
        if fitted:
            # for the moment we dont return levels for fitted values in
            # forecast mode
            fitted_vals = np.full(
                (self.data.shape[0], 1 + cuts[-1]), np.nan, dtype=self.data.dtype
            )
            if self.data.ndim == 1:
                fitted_vals[:, 0] = self.data
            else:
                fitted_vals[:, 0] = self.data[:, 0]
        iterable = tqdm(
            enumerate(self), disable=(not verbose), total=len(self), desc="Forecast"
        )
        times = {repr(m): 0.0 for m in models}
        for i, grp in iterable:
            y_train = grp[:, 0] if grp.ndim == 2 else grp
            X_train = grp[:, 1:] if (grp.ndim == 2 and grp.shape[1] > 1) else None
            if X is not None:
                X_f = X[i]
            else:
                X_f = None
            cols = []
            cols_fitted = []
            for i_model, model in enumerate(models):
                has_level = has_level_models[i_model]
                kwargs = {}
                if has_level:
                    kwargs["level"] = level
                start = time.perf_counter()
                try:
                    res_i = model.forecast(
                        h=h, y=y_train, X=X_train, X_future=X_f, fitted=fitted, **kwargs
                    )
                except Exception as error:
                    if fallback_model is not None:
                        res_i = fallback_model.forecast(
                            h=h,
                            y=y_train,
                            X=X_train,
                            X_future=X_f,
                            fitted=fitted,
                            **kwargs,
                        )
                    else:
                        raise error
                times[repr(model)] += time.perf_counter() - start
                cols_m = [
                    key
                    for key in res_i.keys()
                    if any(key.startswith(m) for m in matches)
                ]
                fcsts_i = np.vstack([res_i[key] for key in cols_m]).T
                cols_m = [
                    f"{repr(model)}" if col == "mean" else f"{repr(model)}-{col}"
                    for col in cols_m
                ]
                if fcsts_i.ndim == 1:
                    fcsts_i = fcsts_i[:, None]
                fcsts[i * h : (i + 1) * h, cuts[i_model] : cuts[i_model + 1]] = fcsts_i
                cols += cols_m
                if fitted:
                    cols_m_fitted = [
                        key
                        for key in res_i.keys()
                        if any(key.startswith(m) for m in matches_fitted)
                    ]
                    fitted_i = np.vstack([res_i[key] for key in cols_m_fitted]).T
                    cols_m_fitted = [
                        (
                            f"{repr(model)}"
                            if col == "fitted"
                            else f"{repr(model)}-{col.replace('fitted-', '')}"
                        )
                        for col in cols_m_fitted
                    ]
                    fitted_vals[
                        self.indptr[i] : self.indptr[i + 1],
                        (cuts[i_model] + 1) : (cuts[i_model + 1] + 1),
                    ] = fitted_i
                    cols_fitted += cols_m_fitted
        result = {"forecasts": fcsts, "cols": cols, "times": times}
        if fitted:
            result["fitted"] = {"values": fitted_vals}
            result["fitted"]["cols"] = [target_col] + cols_fitted
        return result

    def cross_validation(
        self,
        models,
        h,
        test_size,
        fallback_model=None,
        step_size=1,
        input_size=None,
        fitted=False,
        level=tuple(),
        refit=True,
        verbose=False,
        target_col="y",
    ):
        # output of size: (ts, window, h)
        if (test_size - h) % step_size:
            raise Exception("`test_size - h` should be module `step_size`")
        n_windows = int((test_size - h) / step_size) + 1
        n_models = len(models)
        cuts, has_level_models = self._get_cols(
            models=models, attr="forecast", h=h, X=None, level=level
        )
        # first column of out is the actual y
        out = np.full(
            (self.n_groups, n_windows, h, 1 + cuts[-1]), np.nan, dtype=self.data.dtype
        )
        if fitted:
            fitted_vals = np.full(
                (self.data.shape[0], n_windows, n_models + 1),
                np.nan,
                dtype=self.data.dtype,
            )
            fitted_idxs = np.full((self.data.shape[0], n_windows), False, dtype=bool)
            last_fitted_idxs = np.full_like(fitted_idxs, False, dtype=bool)
        matches = ["mean", "lo", "hi"]
        steps = list(range(-test_size, -h + 1, step_size))
        for i_ts, grp in enumerate(self):
            iterable = tqdm(
                enumerate(steps, start=0),
                desc=f"Cross Validation Time Series {i_ts + 1}",
                disable=(not verbose),
                total=len(steps),
            )
            fitted_models = [None for _ in range(n_models)]
            for i_window, cutoff in iterable:
                should_fit = i_window == 0 or (refit > 0 and i_window % refit == 0)
                end_cutoff = cutoff + h
                in_size_disp = cutoff if input_size is None else input_size
                y = grp[(cutoff - in_size_disp) : cutoff]
                y_train = y[:, 0] if y.ndim == 2 else y
                X_train = y[:, 1:] if (y.ndim == 2 and y.shape[1] > 1) else None
                y_test = grp[cutoff:] if end_cutoff == 0 else grp[cutoff:end_cutoff]
                X_future = (
                    y_test[:, 1:]
                    if (y_test.ndim == 2 and y_test.shape[1] > 1)
                    else None
                )
                out[i_ts, i_window, :, 0] = y_test[:, 0] if y.ndim == 2 else y_test
                if fitted:
                    fitted_vals[self.indptr[i_ts] : self.indptr[i_ts + 1], i_window, 0][
                        (cutoff - in_size_disp) : cutoff
                    ] = y_train
                    fitted_idxs[self.indptr[i_ts] : self.indptr[i_ts + 1], i_window][
                        (cutoff - in_size_disp) : cutoff
                    ] = True
                    last_fitted_idxs[
                        self.indptr[i_ts] : self.indptr[i_ts + 1], i_window
                    ][cutoff - 1] = True
                cols = [target_col]
                for i_model, model in enumerate(models):
                    has_level = has_level_models[i_model]
                    kwargs = {}
                    if has_level:
                        kwargs["level"] = level
                    # this is implemented like this because not all models have a forward method
                    # so we can't do fit + forward
                    if refit is True:
                        forecast_kwargs = dict(
                            h=h,
                            y=y_train,
                            X=X_train,
                            X_future=X_future,
                            fitted=fitted,
                            **kwargs,
                        )
                        try:
                            res_i = model.forecast(**forecast_kwargs)
                        except Exception as error:
                            if fallback_model is None:
                                raise error
                            res_i = fallback_model.forecast(**forecast_kwargs)
                    else:
                        if should_fit:
                            try:
                                fitted_models[i_model] = model.fit(y=y_train, X=X_train)
                            except Exception as error:
                                if fallback_model is None:
                                    raise error
                                fitted_models[i_model] = fallback_model.new().fit(
                                    y=y_train, X=X_train
                                )
                        res_i = fitted_models[i_model].forward(
                            h=h,
                            y=y_train,
                            X=X_train,
                            X_future=X_future,
                            fitted=fitted,
                            **kwargs,
                        )
                    cols_m = [
                        key
                        for key in res_i.keys()
                        if any(key.startswith(m) for m in matches)
                    ]
                    fcsts_i = np.vstack([res_i[key] for key in cols_m]).T
                    cols_m = [
                        f"{repr(model)}" if col == "mean" else f"{repr(model)}-{col}"
                        for col in cols_m
                    ]
                    out[
                        i_ts, i_window, :, (1 + cuts[i_model]) : (1 + cuts[i_model + 1])
                    ] = fcsts_i
                    if fitted:
                        fitted_vals[
                            self.indptr[i_ts] : self.indptr[i_ts + 1],
                            i_window,
                            i_model + 1,
                        ][(cutoff - in_size_disp) : cutoff] = res_i["fitted"]
                    cols += cols_m
        result = {"forecasts": out.reshape(-1, 1 + cuts[-1]), "cols": cols}
        if fitted:
            result["fitted"] = {
                "values": fitted_vals,
                "idxs": fitted_idxs,
                "last_idxs": last_fitted_idxs,
                "cols": [target_col] + [repr(model) for model in models],
            }
        return result

    def take(self, idxs):
        data, indptr = super().take(idxs)
        return GroupedArray(data, indptr)

    def split(self, n_chunks):
        n_chunks = min(n_chunks, self.n_groups)
        return [
            self.take(idxs) for idxs in np.array_split(range(self.n_groups), n_chunks)
        ]

    def split_fm(self, fm, n_chunks):
        return [
            fm[idxs]
            for idxs in np.array_split(range(self.n_groups), n_chunks)
            if idxs.size
        ]

    @_controller.wrap(limits=1)
    def _single_threaded_fit(self, models, fallback_model=None):
        return self.fit(models=models, fallback_model=fallback_model)

    @_controller.wrap(limits=1)
    def _single_threaded_predict(self, fm, h, X=None, level=tuple()):
        return self.predict(fm=fm, h=h, X=X, level=level)

    @_controller.wrap(limits=1)
    def _single_threaded_fit_predict(self, models, h, X=None, level=tuple()):
        return self.fit_predict(models=models, h=h, X=X, level=level)

    @_controller.wrap(limits=1)
    def _single_threaded_forecast(
        self,
        models,
        h,
        fallback_model=None,
        fitted=False,
        X=None,
        level=tuple(),
        verbose=False,
        target_col="y",
    ):
        return self.forecast(
            models=models,
            h=h,
            fallback_model=fallback_model,
            fitted=fitted,
            X=X,
            level=level,
            verbose=verbose,
            target_col=target_col,
        )

    @_controller.wrap(limits=1)
    def _single_threaded_cross_validation(
        self,
        models,
        h,
        test_size,
        fallback_model=None,
        step_size=1,
        input_size=None,
        fitted=False,
        level=tuple(),
        refit=True,
        verbose=False,
        target_col="y",
    ):
        return self.cross_validation(
            models=models,
            h=h,
            test_size=test_size,
            fallback_model=fallback_model,
            step_size=step_size,
            input_size=input_size,
            fitted=fitted,
            level=level,
            refit=refit,
            verbose=verbose,
            target_col=target_col,
        )

    def simulate(
        self,
        h,
        n_paths,
        models,
        X=None,
        seed=None,
        seeds=None,
        error_distribution="normal",
        error_params=None,
    ):
        n_groups = self.n_groups
        n_models = len(models)
        if seeds is None and seed is not None:
            np.random.seed(seed)
            seeds = np.random.randint(0, 2**31, size=n_groups)
        out = np.full(
            (n_groups * n_paths * h, n_models), fill_value=np.nan, dtype=self.data.dtype
        )
        for i_model, model in enumerate(models):
            for i, grp in enumerate(self):
                y = grp[:, 0] if grp.ndim == 2 else grp
                X_in = grp[:, 1:] if (grp.ndim == 2 and grp.shape[1] > 1) else None
                if X is not None:
                    X_future = X[i]
                else:
                    X_future = None

                group_seed = seeds[i] if seeds is not None else None
                paths = model.simulate(
                    h=h,
                    n_paths=n_paths,
                    y=y,
                    X=X_in,
                    X_future=X_future,
                    seed=group_seed,
                    error_distribution=error_distribution,
                    error_params=error_params,
                )
                out[i * n_paths * h : (i + 1) * n_paths * h, i_model] = paths.flatten()
        return {"forecasts": out, "cols": [repr(m) for m in models]}

    @_controller.wrap(limits=1)
    def _single_threaded_simulate(
        self,
        h,
        n_paths,
        models,
        X=None,
        seed=None,
        seeds=None,
        error_distribution="normal",
        error_params=None,
    ):
        return self.simulate(
            h=h,
            n_paths=n_paths,
            models=models,
            X=X,
            seed=seed,
            seeds=seeds,
            error_distribution=error_distribution,
            error_params=error_params,
        )


def _get_n_jobs(n_groups, n_jobs):
    if n_jobs == -1 or (n_jobs is None):
        actual_n_jobs = os.cpu_count()
    else:
        actual_n_jobs = n_jobs
    return min(n_groups, actual_n_jobs)


class _StatsForecast:
    """The `StatsForecast` class allows you to efficiently fit multiple `StatsForecast` models
    for large sets of time series. It operates on a DataFrame `df` with at least three columns:
    ids, times, and targets.

    The class has a memory-efficient `StatsForecast.forecast` method that avoids storing partial
    model outputs, while the `StatsForecast.fit` and `StatsForecast.predict` methods with the
    Scikit-learn interface store the fitted models.

    The `StatsForecast` class offers parallelization utilities with Dask, Spark, and Ray back-ends.
    See distributed computing example [here](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray).
    """

    def __init__(
        self,
        models: List[Any],
        freq: Union[str, int],
        n_jobs: int = 1,
        fallback_model: Optional[Any] = None,
        verbose: bool = False,
        distribution: Optional[Union[str, Any]] = None,
    ):
        """Initialize StatsForecast with models and configuration.

        The StatsForecast class enables efficient fitting and forecasting of multiple
        statistical models across large sets of time series. It provides memory-efficient
        methods, parallel processing support, and integrates with distributed computing
        frameworks like Dask, Spark, and Ray.

        Args:
            models (List[Any]): List of instantiated StatsForecast model objects.
                Each model should implement the forecast interface. Models must have
                unique names, which can be set using the `alias` parameter.
            freq (str or int): Frequency of the time series data. Must be a valid
                pandas or polars offset alias (e.g., 'D' for daily, 'M' for monthly,
                'H' for hourly), or an integer representing the number of observations
                per cycle.
            n_jobs (int, optional): Number of jobs to use for parallel processing.
                Use -1 to utilize all available CPU cores.
            fallback_model (Any, optional): Model to use when a primary model fails during
                fitting or forecasting. Only works with the `forecast` and `cross_validation`
                methods. If None, exceptions from failing models will be raised.
            verbose (bool, optional): If True, displays TQDM progress bar during single-job
                execution (when n_jobs=1).
            distribution (str or Distribution, optional): Observation distribution to use
                for all models that support it. Accepts the same values as individual model
                ``distribution`` parameters (e.g. ``"gaussian"``, ``"poisson"``,
                ``"negbin"``, ``"student_t"``).  When set, this overrides each supporting
                model's own ``distribution`` setting.  Models that do not expose a
                ``distribution`` attribute are left unchanged.

        """
        self.models = models
        if distribution is not None:
            dist = _get_distribution(distribution)
            for model in self.models:
                if hasattr(model, "distribution"):
                    model.distribution = dist
                    model._dist_ = dist
        self._validate_model_names()
        self.freq = freq
        self.n_jobs = n_jobs
        self.fallback_model = fallback_model
        self.verbose = verbose

    def _validate_model_names(self):
        # Some test models don't have alias
        names = [getattr(model, "alias", lambda: None) for model in self.models]
        names = [x for x in names if x is not None]
        if len(names) != len(set(names)):
            raise ValueError(
                "Model names must be unique. You can use `alias` to set a unique name for each model."
            )

    def _prepare_fit(
        self,
        df: DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        df = ensure_time_dtype(df, time_col)
        validate_freq(df[time_col], self.freq)
        self.uids, last_times, data, indptr, sort_idxs = ufp.process_df(
            df, id_col, time_col, target_col
        )
        if isinstance(df, pd.DataFrame):
            self.last_dates = pd.Index(last_times, name=time_col)
        else:
            self.last_dates = pl_Series(last_times)
        self.ga = GroupedArray(data, indptr)
        self.og_dates = df[time_col].to_numpy()
        if sort_idxs is not None:
            self.og_dates = self.og_dates[sort_idxs]
        self.n_jobs = _get_n_jobs(len(self.ga), self.n_jobs)
        self.df_constructor = type(df)
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._exog = [c for c in df.columns if c not in (id_col, time_col, target_col)]

    def _validate_sizes_for_prediction_intervals(
        self,
        prediction_intervals: Optional[ConformalIntervals],
        offset: int = 0,
    ) -> None:
        if prediction_intervals is None:
            return
        sizes = np.diff(self.ga.indptr) - offset
        # the absolute minimum requires two windows
        min_samples = 2 * prediction_intervals.h + 1
        if np.any(sizes < min_samples):
            raise ValueError(
                f"Minimum samples for computing prediction intervals are {min_samples + offset:,}, "
                "some series have less. Please remove them or adjust the horizon."
            )
        # required samples for current configuration
        required_samples = prediction_intervals.n_windows * prediction_intervals.h + 1
        if np.any(sizes < required_samples):
            warnings.warn(
                f"Prediction intervals settings require at least {required_samples + offset:,} samples, "
                "some series have less and will use less windows."
            )

    def _set_prediction_intervals(
        self, prediction_intervals: Optional[ConformalIntervals]
    ) -> None:
        for model in self.models:
            interval = getattr(model, "prediction_intervals", None)
            if interval is None:
                setattr(model, "prediction_intervals", prediction_intervals)

    def fit(
        self,
        df: DataFrame,
        prediction_intervals: Optional[ConformalIntervals] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        """Fit statistical models to time series data.

        Fits all models specified in the constructor to each time series in the input
        DataFrame. The fitted models are stored internally and can be used later with
        the `predict` method. This follows the scikit-learn fit/predict interface.

        Args:
            df (DataFrame): Input DataFrame containing time series data.
                Must have columns for series identifiers, timestamps, and target values.
                Can optionally include exogenous features.
            prediction_intervals (ConformalIntervals, optional): Configuration for
                calibrating prediction intervals using Conformal Prediction. If provided,
                the models will be prepared to generate prediction intervals.
            id_col (str, optional): Name of the column containing unique identifiers for
                each time series.
            time_col (str, optional): Name of the column containing timestamps or time
                indices. Values can be timestamps (datetime) or integers.
            target_col (str, optional): Name of the column containing the target variable
                to forecast.

        Returns:
            StatsForecast (StatsForecast): Returns self with fitted models stored in the `fitted_` attribute.
                This allows for method chaining.
        """
        self._prepare_fit(
            df=df, id_col=id_col, time_col=time_col, target_col=target_col
        )
        self._validate_sizes_for_prediction_intervals(prediction_intervals)
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        if self.n_jobs == 1:
            self.fitted_ = self.ga.fit(
                models=self.models, fallback_model=self.fallback_model
            )
        else:
            self.fitted_ = self._fit_parallel()
        return self

    def _make_future_df(self, h: int):
        start_dates = ufp.offset_times(self.last_dates, freq=self.freq, n=1)
        dates = ufp.time_ranges(start_dates, freq=self.freq, periods=h)
        uids = ufp.repeat(self.uids, n=h)
        df = self.df_constructor({self.id_col: uids, self.time_col: dates})
        if isinstance(df, pd.DataFrame):
            df = df.reset_index(drop=True)
        return df

    def _parse_X_level(
        self, h: int, X: Optional[DataFrame], level: Optional[List[int]]
    ):
        if level is None:
            level = []
        if X is None:
            return X, level
        expected_shape = (h * len(self.ga), self.ga.data.shape[1] + 1)
        if X.shape != expected_shape:
            raise ValueError(
                f"Expected X to have shape {expected_shape}, but got {X.shape}"
            )
        processed = ufp.process_df(X, self.id_col, self.time_col, None)
        return GroupedArray(processed.data, processed.indptr), level

    def _validate_exog(self, X_df: Optional[DataFrame] = None) -> None:
        if not any(m.uses_exog for m in self.models) or not self._exog:
            return
        err_msg = (
            f"Models require the following exogenous features {self._exog} "
            "for the forecasting step. Please provide them through `X_df`."
        )
        if X_df is None:
            raise ValueError(err_msg)
        missing_exog = [c for c in self._exog if c not in X_df.columns]
        if missing_exog:
            raise ValueError(err_msg)

    def predict(
        self,
        h: int,
        X_df: Optional[DataFrame] = None,
        level: Optional[List[int]] = None,
    ) -> DataFrame:
        """Generate forecasts using previously fitted models.

        Uses the models fitted via the `fit` method to generate predictions for the
        specified forecast horizon. This follows the scikit-learn fit/predict interface.

        Args:
            h (int): Forecast horizon, the number of time steps ahead to predict.
            X_df (DataFrame, optional): DataFrame containing
                future exogenous variables. Required if any models use exogenous features.
                Must have the same structure as training data and include future values for
                all time series and forecast horizon.
            level (List[float], optional): Confidence levels between 0 and 100 for
                prediction intervals (e.g., [80, 95] for 80% and 95% intervals).
                If provided with models configured for prediction intervals, the output
                will include lower and upper bounds.

        Returns:
            DataFrame with forecasts for each model.
                Contains the series identifiers, future timestamps, and one column per model
                with point predictions. If `level` is specified, includes additional columns
                for prediction interval bounds (e.g., 'model-lo-95', 'model-hi-95').
        """
        if not hasattr(self, "fitted_"):
            raise ValueError("You must call the fit method before calling predict.")
        if (
            any(
                getattr(m, "prediction_intervals", None) is not None
                for m in self.models
            )
            and level is None
        ):
            warnings.warn(
                "Prediction intervals are set but `level` was not provided. "
                "Predictions won't have intervals."
            )
        self._validate_exog(X_df)
        X, level = self._parse_X_level(h=h, X=X_df, level=level)
        if self.n_jobs == 1:
            fcsts, cols = self.ga.predict(fm=self.fitted_, h=h, X=X, level=level)
        else:
            fcsts, cols = self._predict_parallel(h=h, X=X, level=level)
        fcsts_df = self._make_future_df(h=h)
        fcsts_df[cols] = fcsts
        return fcsts_df

    def fit_predict(
        self,
        h: int,
        df: DataFrame,
        X_df: Optional[DataFrame] = None,
        level: Optional[List[int]] = None,
        prediction_intervals: Optional[ConformalIntervals] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> DataFrame:
        """Fit models and generate predictions in a single step.

        Combines the `fit` and `predict` methods in a single operation. The fitted models
        are stored internally in the `fitted_` attribute for later use, making this method
        suitable when you need both training and immediate predictions.

        Args:
            h (int): Forecast horizon, the number of time steps ahead to predict.
            df (DataFrame): Input DataFrame containing time series
                data. Must have columns for series identifiers, timestamps, and target values.
                Can optionally include exogenous features.
            X_df (DataFrame, optional): DataFrame containing
                future exogenous variables. Required if any models use exogenous features.
                Must include future values for all time series and forecast horizon.
            level (List[float], optional): Confidence levels between 0 and 100 for
                prediction intervals (e.g., [80, 95]). Required if `prediction_intervals`
                is specified.
            prediction_intervals (ConformalIntervals, optional): Configuration for
                calibrating prediction intervals using Conformal Prediction.
            id_col (str, optional): Name of the column containing unique identifiers for
                each time series.
            time_col (str, optional): Name of the column containing timestamps or time
                indices. Values can be timestamps (datetime) or integers.
            target_col (str, optional): Name of the column containing the target variable
                to forecast.

        Returns:
            DataFrame with forecasts containing series
                identifiers, future timestamps, and predictions from each model. Includes
                prediction intervals if `level` is specified.
        """
        self._prepare_fit(
            df=df, id_col=id_col, time_col=time_col, target_col=target_col
        )
        self._validate_exog(X_df)
        if prediction_intervals is not None and level is None:
            raise ValueError(
                "You must specify `level` when using `prediction_intervals`"
            )
        self._validate_sizes_for_prediction_intervals(prediction_intervals)
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        X, level = self._parse_X_level(h=h, X=X_df, level=level)
        if self.n_jobs == 1:
            self.fitted_, fcsts, cols = self.ga.fit_predict(
                models=self.models, h=h, X=X, level=level
            )
        else:
            self.fitted_, fcsts, cols = self._fit_predict_parallel(
                h=h, X=X, level=level
            )
        fcsts_df = self._make_future_df(h=h)
        fcsts_df[cols] = fcsts
        return fcsts_df

    def forecast(
        self,
        h: int,
        df: DataFrame,
        X_df: Optional[DataFrame] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
        prediction_intervals: Optional[ConformalIntervals] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> DataFrame:
        """Generate forecasts with memory-efficient model training.

        This is the primary forecasting method that trains models and generates predictions
        without storing fitted model objects. It is more memory-efficient than `fit_predict`
        when you don't need to inspect or reuse the fitted models. Models are trained and
        used for forecasting within each time series, then discarded.

        Args:
            h (int): Forecast horizon, the number of time steps ahead to predict.
            df (DataFrame): Input DataFrame containing time series
                data. Must have columns for series identifiers, timestamps, and target values.
                Can optionally include exogenous features for training.
            X_df (DataFrame, optional): DataFrame containing
                future exogenous variables. Required if any models use exogenous features.
                Must include future values for all time series and forecast horizon.
            level (List[float], optional): Confidence levels between 0 and 100 for
                prediction intervals (e.g., [80, 95]).
            fitted (bool, optional): If True, stores in-sample (fitted) predictions which
                can be retrieved using `forecast_fitted_values()`.
            prediction_intervals (ConformalIntervals, optional): Configuration for
                calibrating prediction intervals using Conformal Prediction.
            id_col (str, optional): Name of the column containing unique identifiers for
                each time series.
            time_col (str, optional): Name of the column containing timestamps or time
                indices. Values can be timestamps (datetime) or integers.
            target_col (str, optional): Name of the column containing the target variable
                to forecast.

        Returns:
            DataFrame with forecasts containing series
                identifiers, future timestamps, and predictions from each model. Includes
                prediction intervals if `level` is specified.
        """
        self.__dict__.pop("fcst_fitted_values_", None)
        self._prepare_fit(
            df=df, id_col=id_col, time_col=time_col, target_col=target_col
        )
        self._validate_exog(X_df)
        self._validate_sizes_for_prediction_intervals(prediction_intervals)
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        X, level = self._parse_X_level(h=h, X=X_df, level=level)
        if self.n_jobs == 1:
            res_fcsts = self.ga.forecast(
                models=self.models,
                h=h,
                fallback_model=self.fallback_model,
                fitted=fitted,
                X=X,
                level=level,
                verbose=self.verbose,
                target_col=target_col,
            )
        else:
            res_fcsts = self._forecast_parallel(
                h=h,
                fitted=fitted,
                X=X,
                level=level,
                target_col=target_col,
            )
        if fitted:
            self.fcst_fitted_values_ = res_fcsts["fitted"]
        fcsts = res_fcsts["forecasts"]
        cols = res_fcsts["cols"]
        fcsts_df = self._make_future_df(h=h)
        fcsts_df[cols] = fcsts
        self.forecast_times_ = res_fcsts["times"]
        return fcsts_df

    def _simulate_parallel(
        self, h, n_paths, X, seed, error_distribution="normal", error_params=None
    ):
        gas, Xs = self._get_gas_Xs(X=X, tasks_per_job=1)

        # Pre-calculate seeds for each group to ensure consistency across models
        # and different seeds across groups even in parallel
        if seed is not None:
            np.random.seed(seed)
            all_seeds = np.random.randint(0, 2**31, size=self.ga.n_groups)
        else:
            all_seeds = None

        results = [None] * len(gas)
        cumsum_groups = np.cumsum([0] + [ga.n_groups for ga in gas])

        with ProcessPoolExecutor(self.n_jobs) as executor:
            future2pos = {
                executor.submit(
                    ga._single_threaded_simulate,
                    h=h,
                    n_paths=n_paths,
                    models=self.models,
                    X=X,
                    seed=None,  # Already handled by passing seeds
                    seeds=all_seeds[cumsum_groups[i] : cumsum_groups[i + 1]]
                    if all_seeds is not None
                    else None,
                    error_distribution=error_distribution,
                    error_params=error_params,
                ): i
                for i, (ga, X) in enumerate(zip(gas, Xs))
            }
            iterable = tqdm(
                as_completed(future2pos),
                disable=not self.verbose,
                total=len(future2pos),
                desc="Simulate",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}{postfix}]",
            )
            for future in iterable:
                i = future2pos[future]
                results[i] = future.result()

        return {
            "cols": results[0]["cols"],
            "forecasts": np.vstack([r["forecasts"] for r in results]),
        }

    def simulate(
        self,
        h: int,
        df: DataFrame,
        X_df: Optional[DataFrame] = None,
        n_paths: int = 1,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        seed: Optional[int] = None,
        error_distribution: str = "normal",
        error_params: Optional[Dict] = None,
    ) -> DataFrame:
        """Generate sample trajectories (simulated paths).

        This method generates `n_paths` simulated trajectories for each time series
        in the input DataFrame. It's useful for scenario planning and risk analysis.

        Args:
            h (int): Forecast horizon.
            df (DataFrame): Input DataFrame containing time series data.
            X_df (DataFrame, optional): Future exogenous variables.
            n_paths (int): Number of paths to simulate.
            id_col (str): Name of the column containing unique identifiers.
            time_col (str): Name of the column containing timestamps.
            target_col (str): Name of the column containing target values.
            seed (int, optional): Random seed for reproducibility.
            error_distribution (str, optional): Error distribution for the simulation.
                Options: 'normal', 't', 'bootstrap', 'laplace', 'skew-normal', 'ged'.
            error_params (dict, optional): Distribution-specific parameters.

        Returns:
            DataFrame with simulated paths, including a `sample_id` column.
        """
        self._prepare_fit(df, id_col, time_col, target_col)
        total_points = len(self.uids) * n_paths * h
        if total_points > 100_000:
            warnings.warn(
                f"Generating {total_points:,} simulation points. "
                "Large simulations may consume significant memory and time."
            )
        self._validate_exog(X_df)
        X, _ = self._parse_X_level(h, X_df, None)

        if self.n_jobs == 1:
            res_sim = self.ga.simulate(
                h=h,
                n_paths=n_paths,
                models=self.models,
                X=X,
                seed=seed,
                error_distribution=error_distribution,
                error_params=error_params,
            )
        else:
            res_sim = self._simulate_parallel(
                h=h,
                n_paths=n_paths,
                X=X,
                seed=seed,
                error_distribution=error_distribution,
                error_params=error_params,
            )

        fcsts = res_sim["forecasts"]
        cols = res_sim["cols"]

        uids = np.repeat(self.uids, n_paths * h)
        dates = np.tile(
            self._make_future_df(h)[self.time_col].to_numpy().reshape(-1, h),
            (1, n_paths),
        ).flatten()

        res_dict = {
            self.id_col: uids,
            self.time_col: dates,
            "sample_id": np.tile(np.repeat(np.arange(n_paths), h), len(self.uids)),
        }
        for i, col in enumerate(cols):
            res_dict[col] = fcsts[:, i]

        return self.df_constructor(res_dict)

    def forecast_fitted_values(self):
        """Retrieve in-sample predictions from the forecast method.

        Returns the fitted (in-sample) predictions generated during the last call to
        `forecast()`. These are the model's predictions on the training data, useful
        for assessing model fit quality and identifying patterns in residuals.

        Returns:
            pandas.DataFrame or polars.DataFrame: DataFrame containing in-sample predictions
                with columns for series identifiers, timestamps, target values, and fitted
                predictions from each model. Includes prediction intervals if they were
                requested during forecasting.
        """
        if not hasattr(self, "fcst_fitted_values_"):
            raise Exception("Please run `forecast` method using `fitted=True`")
        cols = self.fcst_fitted_values_["cols"]
        df = self.df_constructor(
            {
                self.id_col: ufp.repeat(self.uids, np.diff(self.ga.indptr)),
                self.time_col: self.og_dates,
            }
        )
        df[cols] = self.fcst_fitted_values_["values"]
        if isinstance(df, pd.DataFrame):
            df = df.reset_index(drop=True)
        return df

    def cross_validation(
        self,
        h: int,
        df: DataFrame,
        n_windows: int = 1,
        step_size: int = 1,
        test_size: Optional[int] = None,
        input_size: Optional[int] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
        refit: Union[bool, int] = True,
        prediction_intervals: Optional[ConformalIntervals] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> DataFrame:
        """Perform temporal cross-validation for model evaluation.

        Evaluates model performance across multiple time windows using a time series
        cross-validation approach. This method trains models on expanding or rolling
        windows and generates forecasts for each validation period, providing robust
        assessment of forecast accuracy and generalization.

        Args:
            h (int): Forecast horizon for each validation window.
            df (pandas.DataFrame or polars.DataFrame): Input DataFrame containing time series
                data with columns for series identifiers, timestamps, and target values.
            n_windows (int, optional): Number of validation windows to create. Cannot be
                specified together with `test_size`.
            step_size (int, optional): Number of time steps between consecutive validation
                windows. Smaller values create overlapping windows.
            test_size (int, optional): Total size of the test period. If provided, `n_windows`
                is computed automatically. Overrides `n_windows` if specified.
            input_size (int, optional): Maximum number of training observations to use for
                each window. If None, uses expanding windows with all available history.
                If specified, uses rolling windows of fixed size.
            level (List[float], optional): Confidence levels between 0 and 100 for
                prediction intervals (e.g., [80, 95]).
            fitted (bool, optional): If True, stores in-sample predictions for each window,
                accessible via `cross_validation_fitted_values()`.
            refit (bool or int, optional): Controls model refitting frequency. If True,
                refits models for every window. If False, fits once and uses the forward
                method. If an integer n, refits every n windows. Models must implement the
                `forward` method when refit is not True.
            prediction_intervals (ConformalIntervals, optional): Configuration for
                calibrating prediction intervals using Conformal Prediction. Requires
                `level` to be specified.
            id_col (str, optional): Name of the column containing unique identifiers for
                each time series.
            time_col (str, optional): Name of the column containing timestamps or time
                indices.
            target_col (str, optional): Name of the column containing the target variable.

        Returns:
            DataFrame with cross-validation results
                including series identifiers, cutoff dates (last training observation),
                forecast dates, actual values, and predictions from each model for all windows.
        """
        if n_windows is None and test_size is None:
            raise ValueError("you must define `n_windows` or `test_size`")
        if test_size is None:
            test_size = h + step_size * (n_windows - 1)
        if prediction_intervals is not None and level is None:
            raise ValueError(
                "You must specify `level` when using `prediction_intervals`"
            )
        if refit != True:
            no_forward = [m for m in self.models if not hasattr(m, "forward")]
            if no_forward:
                raise ValueError(
                    "Can only use integer refit or refit=False with models that implement the forward method. "
                    f"The following models do not implement the forward method: {no_forward}."
                )
            if self.fallback_model is not None and not hasattr(
                self.fallback_model, "forward"
            ):
                raise ValueError(
                    "Can only use integer refit or refit=False with a fallback model that implements the forward method."
                )
        self.__dict__.pop("cv_fitted_values_", None)
        self._prepare_fit(
            df=df, id_col=id_col, time_col=time_col, target_col=target_col
        )
        series_sizes = np.diff(self.ga.indptr)
        short_series = series_sizes <= test_size
        if short_series.any():
            short_ids = self.uids[short_series].to_numpy().tolist()
            raise ValueError(
                f"The following series are too short for the cross validation settings: {reprlib.repr(short_ids)}\n"
                "Please remove these series or change the settings, e.g. reducing the horizon or the number of windows."
            )
        self._validate_sizes_for_prediction_intervals(
            prediction_intervals=prediction_intervals, offset=test_size
        )
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        _, level = self._parse_X_level(h=h, X=None, level=level)
        if self.n_jobs == 1:
            res_fcsts = self.ga.cross_validation(
                models=self.models,
                h=h,
                test_size=test_size,
                fallback_model=self.fallback_model,
                step_size=step_size,
                input_size=input_size,
                fitted=fitted,
                level=level,
                verbose=self.verbose,
                refit=refit,
                target_col=target_col,
            )
        else:
            res_fcsts = self._cross_validation_parallel(
                h=h,
                test_size=test_size,
                step_size=step_size,
                input_size=input_size,
                fitted=fitted,
                level=level,
                refit=refit,
                target_col=target_col,
            )
        if fitted:
            self.cv_fitted_values_ = res_fcsts["fitted"]
            self.n_cv_ = n_windows
        fcsts_df = ufp.cv_times(
            times=self.og_dates,
            uids=self.uids,
            indptr=self.ga.indptr,
            h=h,
            test_size=test_size,
            step_size=step_size,
            id_col=id_col,
            time_col=time_col,
        )
        # the cv_times is sorted by window and then id
        fcsts_df = ufp.sort(fcsts_df, [id_col, "cutoff", time_col])
        fcsts_df = ufp.assign_columns(
            fcsts_df, res_fcsts["cols"], res_fcsts["forecasts"]
        )
        return fcsts_df

    def cross_validation_fitted_values(self) -> DataFrame:
        """Retrieve in-sample predictions from cross-validation.

        Returns the fitted (in-sample) predictions for each cross-validation window.
        These are the model's predictions on the training data for each window,
        useful for analyzing how model fit changes across different training periods.

        Returns:
            pandas.DataFrame or polars.DataFrame: DataFrame containing in-sample predictions
                for each cross-validation window. Includes columns for series identifiers,
                timestamps, cutoff dates (last training observation of each window), actual
                values, and fitted predictions from each model.
        """
        if not hasattr(self, "cv_fitted_values_"):
            raise Exception("Please run `cross_validation` method using `fitted=True`")
        idxs = self.cv_fitted_values_["idxs"].flatten(order="F")
        train_uids = ufp.repeat(self.uids, np.diff(self.ga.indptr))
        cv_uids = ufp.vertical_concat([train_uids for _ in range(self.n_cv_)])
        used_uids = ufp.take_rows(cv_uids, idxs)
        dates = np.tile(self.og_dates, self.n_cv_)[idxs]
        cutoffs_mask = self.cv_fitted_values_["last_idxs"].flatten(order="F")[idxs]
        cutoffs_sizes = np.diff(np.append(0, np.where(cutoffs_mask)[0] + 1))
        cutoffs = np.repeat(dates[cutoffs_mask], cutoffs_sizes)
        df = self.df_constructor(
            {
                self.id_col: used_uids,
                self.time_col: dates,
                "cutoff": cutoffs,
            }
        )
        fitted_vals = np.reshape(
            self.cv_fitted_values_["values"],
            (-1, len(self.models) + 1),
            order="F",
        )
        df = ufp.assign_columns(df, self.cv_fitted_values_["cols"], fitted_vals[idxs])
        df = ufp.drop_index_if_pandas(df)
        if isinstance(df, pd.DataFrame):
            df = df.reset_index(drop=True)
        return df

    def _get_pool(self):
        from multiprocessing import Pool

        pool_kwargs = dict()
        return Pool, pool_kwargs

    def _fit_parallel(self):
        gas = self.ga.split(self.n_jobs)
        Pool, pool_kwargs = self._get_pool()
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga in gas:
                future = executor.apply_async(
                    ga._single_threaded_fit, (self.models, self.fallback_model)
                )
                futures.append(future)
            fm = np.vstack([f.get() for f in futures])
        return fm

    def _get_gas_Xs(self, X, tasks_per_job=1):
        n_chunks = min(tasks_per_job * self.n_jobs, self.ga.n_groups)
        gas = self.ga.split(n_chunks)
        if X is not None:
            Xs = X.split(n_chunks)
        else:
            from itertools import repeat

            Xs = repeat(None)
        return gas, Xs

    def _predict_parallel(self, h, X, level):
        # create elements for each core
        gas, Xs = self._get_gas_Xs(X=X)
        fms = self.ga.split_fm(self.fitted_, self.n_jobs)
        Pool, pool_kwargs = self._get_pool()
        # compute parallel forecasts
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga, fm, X_ in zip(gas, fms, Xs):
                future = executor.apply_async(
                    ga._single_threaded_predict,
                    (fm, h, X_, level),
                )
                futures.append(future)
            out = [f.get() for f in futures]
            fcsts, cols = list(zip(*out))
            fcsts = np.vstack(fcsts)
            cols = cols[0]
        return fcsts, cols

    def _fit_predict_parallel(self, h, X, level):
        # create elements for each core
        gas, Xs = self._get_gas_Xs(X=X)
        Pool, pool_kwargs = self._get_pool()
        # compute parallel forecasts
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga, X_ in zip(gas, Xs):
                future = executor.apply_async(
                    ga._single_threaded_fit_predict,
                    (self.models, h, X_, level),
                )
                futures.append(future)
            out = [f.get() for f in futures]
            fm, fcsts, cols = list(zip(*out))
            fm = np.vstack(fm)
            fcsts = np.vstack(fcsts)
            cols = cols[0]
        return fm, fcsts, cols

    def _forecast_parallel(self, h, fitted, X, level, target_col):
        gas, Xs = self._get_gas_Xs(X=X, tasks_per_job=100)
        results = [None] * len(gas)
        with ProcessPoolExecutor(self.n_jobs) as executor:
            future2pos = {
                executor.submit(
                    ga._single_threaded_forecast,
                    h=h,
                    models=self.models,
                    fallback_model=self.fallback_model,
                    fitted=fitted,
                    X=X,
                    level=level,
                    verbose=False,
                    target_col=target_col,
                ): i
                for i, (ga, X) in enumerate(zip(gas, Xs))
            }
            iterable = tqdm(
                as_completed(future2pos),
                disable=not self.verbose,
                total=len(future2pos),
                desc="Forecast",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}{postfix}]",
            )
            for future in iterable:
                i = future2pos[future]
                results[i] = future.result()
        result = {
            "cols": results[0]["cols"],
            "forecasts": np.vstack([r["forecasts"] for r in results]),
            "times": {
                m: sum(r["times"][m] for r in results)
                for m in [repr(m) for m in self.models]
            },
        }
        if fitted:
            result["fitted"] = {
                "cols": results[0]["fitted"]["cols"],
                "values": np.vstack([r["fitted"]["values"] for r in results]),
            }
        return result

    def _cross_validation_parallel(
        self, h, test_size, step_size, input_size, fitted, level, refit, target_col
    ):
        # create elements for each core
        gas = self.ga.split(self.n_jobs)
        Pool, pool_kwargs = self._get_pool()
        # compute parallel forecasts
        result = {}
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga in gas:
                future = executor.apply_async(
                    ga._single_threaded_cross_validation,
                    tuple(),
                    dict(
                        models=self.models,
                        h=h,
                        test_size=test_size,
                        fallback_model=self.fallback_model,
                        step_size=step_size,
                        input_size=input_size,
                        fitted=fitted,
                        level=level,
                        refit=refit,
                        verbose=self.verbose,
                        target_col=target_col,
                    ),
                )
                futures.append(future)
            out = [f.get() for f in futures]
            fcsts = [d["forecasts"] for d in out]
            fcsts = np.vstack(fcsts)
            cols = out[0]["cols"]
            result["forecasts"] = fcsts
            result["cols"] = cols
            if fitted:
                result["fitted"] = {}
                result["fitted"]["values"] = np.concatenate(
                    [d["fitted"]["values"] for d in out]
                )
                for key in ["last_idxs", "idxs"]:
                    result["fitted"][key] = np.concatenate(
                        [d["fitted"][key] for d in out]
                    )
                result["fitted"]["cols"] = out[0]["fitted"]["cols"]
        return result

    @staticmethod
    def plot(
        df: DataFrame,
        forecasts_df: Optional[DataFrame] = None,
        unique_ids: Union[Optional[List[str]], np.ndarray] = None,
        plot_random: bool = True,
        models: Optional[List[str]] = None,
        level: Optional[List[float]] = None,
        max_insample_length: Optional[int] = None,
        plot_anomalies: bool = False,
        engine: str = "matplotlib",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        resampler_kwargs: Optional[Dict] = None,
    ):
        """Visualize time series data with forecasts and prediction intervals.

        Creates plots showing historical data, forecasts, and optional prediction intervals
        for time series. Supports multiple plotting engines and interactive visualization.

        Args:
            df (DataFrame): Input DataFrame containing historical
                time series data with columns for series identifiers, timestamps, and target values.
            forecasts_df (DataFrame, optional): DataFrame with forecast
                results from `forecast()` or `cross_validation()`. Should contain series identifiers,
                timestamps, and model predictions.
            unique_ids (List[str] or numpy.ndarray, optional): Specific series identifiers to plot.
                If None and `plot_random` is True, series are selected randomly.
            plot_random (bool, optional): Whether to randomly select series to plot when
                `unique_ids` is not specified.
            models (List[str], optional): Names of specific models to include in the plot.
                If None, plots all models present in `forecasts_df`.
            level (List[float], optional): Confidence levels to plot as shaded regions around
                forecasts (e.g., [80, 95]). Only applicable if prediction intervals are present
                in `forecasts_df`.
            max_insample_length (int, optional): Maximum number of historical observations to
                display. Useful for focusing on recent history when series are long.
            plot_anomalies (bool, optional): If True, highlights observations that fall outside
                prediction intervals as anomalies.
            engine (str, optional): Plotting library to use. Options are 'matplotlib' (static plots),
                'plotly' (interactive plots), or 'plotly-resampler' (interactive with downsampling
                for large datasets).
            id_col (str, optional): Name of the column containing series identifiers.
            time_col (str, optional): Name of the column containing timestamps.
            target_col (str, optional): Name of the column containing the target variable.
            resampler_kwargs (Dict, optional): Additional keyword arguments passed to the
                plotly-resampler constructor when `engine='plotly-resampler'`. For further
                customization (e.g., 'show_dash'), call this method, store the returned object,
                and add arguments to its `show_dash` method.

        Returns:
            Plotting object from the selected engine (matplotlib Figure, plotly Figure, or
            FigureResampler object), which can be further customized or displayed.
        """
        from utilsforecast.plotting import plot_series

        df = ensure_time_dtype(df, time_col)
        if forecasts_df is not None:
            forecasts_df = ensure_time_dtype(forecasts_df, time_col)
        return plot_series(
            df=df,
            forecasts_df=forecasts_df,
            ids=unique_ids,
            plot_random=plot_random,
            models=models,
            level=level,
            max_insample_length=max_insample_length,
            plot_anomalies=plot_anomalies,
            engine=engine,
            resampler_kwargs=resampler_kwargs,
            palette="tab20b",
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

    def save(
        self,
        path: Optional[Union[Path, str]] = None,
        max_size: Optional[str] = None,
        trim: bool = False,
    ):
        """Save the StatsForecast instance to disk using pickle.

        Serializes the StatsForecast object including all fitted models and configuration
        to a file for later use. The saved object can be loaded with the `load()` method
        to restore the exact state for making predictions.

        Args:
            path (str or pathlib.Path, optional): File path where the object will be saved.
                If None, creates a filename in the current directory using the format
                'StatsForecast_YYYY-MM-DD_HH-MM-SS.pkl' with the current UTC timestamp.
            max_size (str, optional): Maximum allowed size for the serialized object.
                Should be specified as a number followed by a unit: 'B', 'KB', 'MB', or 'GB'
                (e.g., '100MB', '1.5GB'). If the object exceeds this size, an OSError is raised.
            trim (bool, optional): If True, removes fitted values from `forecast()` and
                `cross_validation()` before saving to reduce file size. These values are
                not needed for generating new predictions.
        """
        # Will be used to find the size of the fitted models
        # Never expecting anything higher than GB (even that's a lot')
        bytes_hmap = {
            "B": 1,
            "KB": 2**10,
            "MB": 2**20,
            "GB": 2**30,
        }

        # Removing unnecessary attributes
        # @jmoralez decide future implementation
        trim_attr: list = ["fcst_fitted_values_", "cv_fitted_values_"]
        if trim:
            for attr in trim_attr:
                # remove unnecessary attributes here
                self.__dict__.pop(attr, None)

        sf_size = len(pickle.dumps(self))

        if max_size is not None:
            cap_size = self._get_cap_size(max_size, bytes_hmap)
            if sf_size >= cap_size:
                err_messg = "StatsForecast is larger than the specified max_size"
                raise OSError(errno.EFBIG, err_messg)

        converted_size, sf_byte = None, None
        for key in reversed(list(bytes_hmap.keys())):
            x_byte = bytes_hmap[key]
            if sf_size >= x_byte:
                converted_size = sf_size / x_byte
                sf_byte = key
                break

        if converted_size is None or sf_byte is None:
            err_messg = "Internal Error, this shouldn't happen, please open an issue"
            raise RuntimeError(err_messg)

        print(f"Saving StatsForecast object of size {converted_size:.2f}{sf_byte}.")

        if path is None:
            datetime_record = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            path = f"StatsForecast_{datetime_record}.pkl"

        with open(path, "wb") as m_file:
            pickle.dump(self, m_file)
        print("StatsForecast object saved")

    def _get_cap_size(self, max_size, bytes_hmap):
        max_size = max_size.upper().replace(" ", "")
        match = re.match(r"(\d+\.\d+|\d+)(\w+)", max_size)
        if (
            match is None
            or len(match.groups()) < 2
            or match[2] not in bytes_hmap.keys()
        ):
            parsing_error = (
                "Couldn't parse `max_size`, it should be `None`",
                " or a number followed by one of the following units: ['B', 'KB', 'MB', 'GB']",
            )
            raise ValueError(parsing_error)
        else:
            m_size = float(match[1])
            key_ = match[2]
            cap_size = m_size * bytes_hmap[key_]
        return cap_size

    @staticmethod
    def load(path: Union[Path, str]):
        """Load a previously saved StatsForecast instance from disk.

        Deserializes a StatsForecast object that was saved using the `save()` method,
        restoring all fitted models and configuration. The loaded object is ready to
        generate predictions immediately.

        Args:
            path (str or pathlib.Path): File path to the saved StatsForecast pickle file.
                Must point to a file created by the `save()` method.

        Returns:
            StatsForecast: The deserialized StatsForecast instance with all fitted models
                and configuration restored, ready for prediction.
        """
        if not Path(path).exists():
            raise ValueError("Specified path does not exist, check again and retry.")
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return f"StatsForecast(models=[{','.join(map(repr, self.models))}])"


class ParallelBackend:
    def forecast(
        self,
        *,
        models,
        fallback_model,
        freq,
        h,
        df,
        X_df,
        level,
        fitted,
        prediction_intervals,
        id_col,
        time_col,
        target_col,
    ) -> Any:
        model = _StatsForecast(
            models=models,
            freq=freq,
            fallback_model=fallback_model,
        )
        return model.forecast(
            df=df,
            h=h,
            X_df=X_df,
            level=level,
            fitted=fitted,
            prediction_intervals=prediction_intervals,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

    def cross_validation(
        self,
        *,
        df,
        models,
        freq,
        fallback_model,
        h,
        n_windows,
        step_size,
        test_size,
        input_size,
        level,
        refit,
        fitted,
        prediction_intervals,
        id_col,
        time_col,
        target_col,
    ) -> Any:
        model = _StatsForecast(
            models=models,
            freq=freq,
            fallback_model=fallback_model,
        )
        return model.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            test_size=test_size,
            input_size=input_size,
            level=level,
            refit=refit,
            fitted=fitted,
            prediction_intervals=prediction_intervals,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )


@conditional_dispatcher
def make_backend(obj: Any, *args: Any, **kwargs: Any) -> ParallelBackend:
    return ParallelBackend()


class StatsForecast(_StatsForecast):
    def forecast(
        self,
        h: int,
        df: Any,
        X_df: Optional[DataFrame] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
        prediction_intervals: Optional[ConformalIntervals] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> DataFrame:
        """Generate forecasts with memory-efficient model training.

        This is the primary forecasting method that trains models and generates predictions
        without storing fitted model objects. It is more memory-efficient than `fit_predict`
        when you don't need to inspect or reuse the fitted models. Models are trained and
        used for forecasting within each time series, then discarded.

        Args:
            h (int): Forecast horizon, the number of time steps ahead to predict.
            df (DataFrame): Input DataFrame containing time series
                data. Must have columns for series identifiers, timestamps, and target values.
                Can optionally include exogenous features for training.
            X_df (DataFrame, optional): DataFrame containing
                future exogenous variables. Required if any models use exogenous features.
                Must include future values for all time series and forecast horizon.
            level (List[float], optional): Confidence levels between 0 and 100 for
                prediction intervals (e.g., [80, 95]).
            fitted (bool, optional): If True, stores in-sample (fitted) predictions which
                can be retrieved using `forecast_fitted_values()`.
            prediction_intervals (ConformalIntervals, optional): Configuration for
                calibrating prediction intervals using Conformal Prediction.
            id_col (str, optional): Name of the column containing unique identifiers for
                each time series.
            time_col (str, optional): Name of the column containing timestamps or time
                indices. Values can be timestamps (datetime) or integers.
            target_col (str, optional): Name of the column containing the target variable
                to forecast.

        Returns:
            DataFrame with forecasts containing series
                identifiers, future timestamps, and predictions from each model. Includes
                prediction intervals if `level` is specified.
        """
        if prediction_intervals is not None and level is None:
            raise ValueError(
                "You must specify `level` when using `prediction_intervals`"
            )
        if self._is_native(df=df):
            return super().forecast(
                df=df,
                h=h,
                X_df=X_df,
                level=level,
                fitted=fitted,
                prediction_intervals=prediction_intervals,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
        assert df is not None
        engine = make_execution_engine(infer_by=[df])
        self._backend = make_backend(engine)
        return self._backend.forecast(
            models=self.models,
            fallback_model=self.fallback_model,
            freq=self.freq,
            df=df,
            h=h,
            X_df=X_df,
            level=level,
            fitted=fitted,
            prediction_intervals=prediction_intervals,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

    def simulate(
        self,
        h: int,
        df: Any,
        X_df: Optional[DataFrame] = None,
        n_paths: int = 1,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        seed: Optional[int] = None,
        error_distribution: str = "normal",
        error_params: Optional[Dict] = None,
    ) -> DataFrame:
        """Generate sample trajectories (simulated paths).

        This method generates `n_paths` simulated trajectories for each time series
        in the input DataFrame. It's useful for scenario planning and risk analysis.

        Args:
            h (int): Forecast horizon, the number of time steps ahead to predict.
            df (DataFrame): Input DataFrame containing time series data. Must have
                columns for series identifiers, timestamps, and target values.
            X_df (DataFrame, optional): DataFrame containing future exogenous variables.
            n_paths (int): Number of paths to simulate.
            id_col (str, optional): Name of the column containing unique identifiers.
            time_col (str, optional): Name of the column containing timestamps.
            target_col (str, optional): Name of the column containing target values.
            seed (int, optional): Random seed for reproducibility.
            error_distribution (str, optional): Error distribution for the simulation.
                Options: 'normal', 't', 'bootstrap', 'laplace', 'skew-normal', 'ged'.
            error_params (dict, optional): Distribution-specific parameters.

        Returns:
            DataFrame with simulated paths, including a `sample_id` column.
        """
        if self._is_native(df=df):
            return super().simulate(
                h=h,
                df=df,
                X_df=X_df,
                n_paths=n_paths,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                seed=seed,
                error_distribution=error_distribution,
                error_params=error_params,
            )
        assert df is not None
        # Distributed simulation not implemented yet, fallback to native if possible or raise
        warnings.warn(
            "Distributed simulation is not yet implemented. Falling back to native execution."
        )
        return super().simulate(
            h=h,
            df=df,
            X_df=X_df,
            n_paths=n_paths,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            seed=seed,
            error_distribution=error_distribution,
            error_params=error_params,
        )

    def forecast_fitted_values(self):
        if hasattr(self, "_backend"):
            res = self._backend.forecast_fitted_values()
        else:
            res = super().forecast_fitted_values()
        return res

    def cross_validation(
        self,
        h: int,
        df: Any,
        n_windows: int = 1,
        step_size: int = 1,
        test_size: Optional[int] = None,
        input_size: Optional[int] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
        refit: Union[bool, int] = True,
        prediction_intervals: Optional[ConformalIntervals] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> DataFrame:
        """Perform temporal cross-validation for model evaluation.

        Evaluates model performance across multiple time windows using a time series
        cross-validation approach. This method trains models on expanding or rolling
        windows and generates forecasts for each validation period, providing robust
        assessment of forecast accuracy and generalization.

        Args:
            h (int): Forecast horizon for each validation window.
            df (DataFrame): Input DataFrame containing time series
                data with columns for series identifiers, timestamps, and target values.
            n_windows (int, optional): Number of validation windows to create. Cannot be
                specified together with `test_size`.
            step_size (int, optional): Number of time steps between consecutive validation
                windows. Smaller values create overlapping windows.
            test_size (int, optional): Total size of the test period. If provided, `n_windows`
                is computed automatically. Overrides `n_windows` if specified.
            input_size (int, optional): Maximum number of training observations to use for
                each window. If None, uses expanding windows with all available history.
                If specified, uses rolling windows of fixed size.
            level (List[float], optional): Confidence levels between 0 and 100 for
                prediction intervals (e.g., [80, 95]).
            fitted (bool, optional): If True, stores in-sample predictions for each window,
                accessible via `cross_validation_fitted_values()`.
            refit (bool or int, optional): Controls model refitting frequency. If True,
                refits models for every window. If False, fits once and uses the forward
                method. If an integer n, refits every n windows. Models must implement the
                `forward` method when refit is not True.
            prediction_intervals (ConformalIntervals, optional): Configuration for
                calibrating prediction intervals using Conformal Prediction. Requires
                `level` to be specified.
            id_col (str, optional): Name of the column containing unique identifiers for
                each time series.
            time_col (str, optional): Name of the column containing timestamps or time
                indices.
            target_col (str, optional): Name of the column containing the target variable.

        Returns:
            DataFrame with cross-validation results
                including series identifiers, cutoff dates (last training observation),
                forecast dates, actual values, and predictions from each model for all windows.
        """
        if self._is_native(df=df):
            return super().cross_validation(
                h=h,
                df=df,
                n_windows=n_windows,
                step_size=step_size,
                test_size=test_size,
                input_size=input_size,
                level=level,
                fitted=fitted,
                refit=refit,
                prediction_intervals=prediction_intervals,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
        assert df is not None
        engine = make_execution_engine(infer_by=[df])
        backend = make_backend(engine)
        return backend.cross_validation(
            df=df,
            models=self.models,
            freq=self.freq,
            fallback_model=self.fallback_model,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            test_size=test_size,
            input_size=input_size,
            level=level,
            refit=refit,
            fitted=fitted,
            prediction_intervals=prediction_intervals,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

    def _is_native(self, df) -> bool:
        engine = try_get_context_execution_engine()
        return engine is None and (
            df is None or isinstance(df, pd.DataFrame) or isinstance(df, pl_DataFrame)
        )
