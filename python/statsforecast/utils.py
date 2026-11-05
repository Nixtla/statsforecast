__all__ = ['AirPassengers', 'AirPassengersDF', 'find_season_length', 'generate_series']


import math
import warnings
from collections import namedtuple
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd
from utilsforecast.compat import DataFrame
from utilsforecast.data import generate_series as utils_generate_series

results = namedtuple("results", "x fn nit simplex")


def generate_series(
    n_series: int,
    freq: str = "D",
    min_length: int = 50,
    max_length: int = 500,
    n_static_features: int = 0,
    equal_ends: bool = False,
    engine: str = "pandas",
    seed: int = 0,
) -> DataFrame:
    """Generate Synthetic Panel Series.

    Generates `n_series` of frequency `freq` of different lengths in the interval [`min_length`, `max_length`].
    If `n_static_features > 0`, then each series gets static features with random values.
    If `equal_ends == True` then all series end at the same date.

    Args:
        n_series (int): Number of series for synthetic panel.
        freq (str, optional): Frequency of the data, 'D' or 'M'. Defaults to 'D'.
        min_length (int, optional): Minimum length of synthetic panel's series. Defaults to 50.
        max_length (int, optional): Maximum length of synthetic panel's series. Defaults to 500.
        n_static_features (int, optional): Number of static exogenous variables for synthetic panel's series. Defaults to 0.
        equal_ends (bool, optional): Series should end in the same date stamp `ds`. Defaults to False.
        engine (str, optional): Output Dataframe type ('pandas' or 'polars'). Defaults to 'pandas'.
        seed (int, optional): Random seed used for generating the data. Defaults to 0.

    Returns:
        DataFrame: Synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous.
    """
    return utils_generate_series(
        n_series=n_series,
        freq=freq,
        min_length=min_length,
        max_length=max_length,
        n_static_features=n_static_features,
        equal_ends=equal_ends,
        engine=engine,
        seed=seed,
    )


AirPassengers = np.array(
    [
        112.0,
        118.0,
        132.0,
        129.0,
        121.0,
        135.0,
        148.0,
        148.0,
        136.0,
        119.0,
        104.0,
        118.0,
        115.0,
        126.0,
        141.0,
        135.0,
        125.0,
        149.0,
        170.0,
        170.0,
        158.0,
        133.0,
        114.0,
        140.0,
        145.0,
        150.0,
        178.0,
        163.0,
        172.0,
        178.0,
        199.0,
        199.0,
        184.0,
        162.0,
        146.0,
        166.0,
        171.0,
        180.0,
        193.0,
        181.0,
        183.0,
        218.0,
        230.0,
        242.0,
        209.0,
        191.0,
        172.0,
        194.0,
        196.0,
        196.0,
        236.0,
        235.0,
        229.0,
        243.0,
        264.0,
        272.0,
        237.0,
        211.0,
        180.0,
        201.0,
        204.0,
        188.0,
        235.0,
        227.0,
        234.0,
        264.0,
        302.0,
        293.0,
        259.0,
        229.0,
        203.0,
        229.0,
        242.0,
        233.0,
        267.0,
        269.0,
        270.0,
        315.0,
        364.0,
        347.0,
        312.0,
        274.0,
        237.0,
        278.0,
        284.0,
        277.0,
        317.0,
        313.0,
        318.0,
        374.0,
        413.0,
        405.0,
        355.0,
        306.0,
        271.0,
        306.0,
        315.0,
        301.0,
        356.0,
        348.0,
        355.0,
        422.0,
        465.0,
        467.0,
        404.0,
        347.0,
        305.0,
        336.0,
        340.0,
        318.0,
        362.0,
        348.0,
        363.0,
        435.0,
        491.0,
        505.0,
        404.0,
        359.0,
        310.0,
        337.0,
        360.0,
        342.0,
        406.0,
        396.0,
        420.0,
        472.0,
        548.0,
        559.0,
        463.0,
        407.0,
        362.0,
        405.0,
        417.0,
        391.0,
        419.0,
        461.0,
        472.0,
        535.0,
        622.0,
        606.0,
        508.0,
        461.0,
        390.0,
        432.0,
    ]
)


AirPassengersDF = pd.DataFrame(
    {
        "unique_id": np.ones(len(AirPassengers)),
        "ds": pd.date_range(
            start="1949-01-01", periods=len(AirPassengers), freq=pd.offsets.MonthEnd()
        ),
        "y": AirPassengers,
    }
)


def _longest_finite_run(y: np.ndarray) -> np.ndarray:
    """Longest stretch of consecutive finite values (R's `na.contiguous`)."""
    finite = np.isfinite(y)
    if finite.all():
        return y
    best_start, best_len, start = 0, 0, None
    for i, ok in enumerate(np.append(finite, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    return y[best_start : best_start + best_len]


def _ar_spectrum(x: np.ndarray, n_freq: int) -> tuple:
    """Spectral density of `x` from an AR fit, as R's `spec.ar` computes it.

    The AR model is fitted by Yule-Walker with its order chosen by AIC, up to
    `min(n - 1, floor(10 * log10(n)))` (R's `ar.yw` defaults). The density is
    evaluated at `n_freq` frequencies from 0 to 0.5 cycles per observation.
    """
    n = x.size
    order_max = int(min(n - 1, np.floor(10 * np.log10(n))))
    xc = x - x.mean()
    acov = np.array([xc[: n - k] @ xc[k:] / n for k in range(order_max + 1)])
    # Levinson-Durbin recursion: AR coefficients and innovation variance per order.
    variances = np.empty(order_max + 1)
    variances[0] = acov[0]
    coefs = [np.zeros(0)]
    phi = np.zeros(0)
    for k in range(1, order_max + 1):
        reflection = (acov[k] - phi @ acov[k - 1 : 0 : -1]) / variances[k - 1]
        phi = np.append(phi - reflection * phi[::-1], reflection)
        variances[k] = variances[k - 1] * (1.0 - reflection**2)
        coefs.append(phi)
    variances = np.maximum(variances, np.finfo(np.float64).eps * acov[0])
    aic = n * np.log(variances) + 2 * np.arange(order_max + 1)
    order = int(np.argmin(aic))
    var_pred = variances[order] * n / (n - (order + 1))
    freq = np.linspace(0.0, 0.5, n_freq)
    if order == 0:
        return freq, np.full(n_freq, var_pred)
    lags = np.arange(1, order + 1)
    cs = np.cos(2 * np.pi * np.outer(freq, lags)) @ coefs[order]
    sn = np.sin(2 * np.pi * np.outer(freq, lags)) @ coefs[order]
    return freq, var_pred / ((1.0 - cs) ** 2 + sn**2)


def find_season_length(
    y: np.ndarray,
    max_season_length: Optional[int] = None,
) -> int:
    """Estimate the season length of a series from its spectrum.

    Port of R's `forecast::findfrequency`: remove a linear trend, fit an AR
    model by Yule-Walker with the order chosen by AIC, and return the period
    of the highest peak of its spectral density, rounded to the nearest
    integer. When there is no clear peak, returns 1 (no seasonality).

    One change from R: the detrended series is scaled to unit variance before
    the peak is tested, so a peak counts when the spectral density there is
    more than 10 times that of white noise with the same variance. R applies
    the same threshold to the unscaled spectrum, which makes the result depend
    on the units of `y` (for example, `findfrequency(AirPassengers / 100)`
    returns 1 instead of 12).

    Args:
        y (np.ndarray): Time series, one-dimensional. Non-finite values are
            handled as in R: only the longest stretch of consecutive finite
            values is used.
        max_season_length (int, optional): Largest season length to consider.
            Peaks at longer periods are ignored, which avoids very large
            values that make seasonal models slow. Defaults to None (no limit).

    Returns:
        int: Estimated season length; 1 when no seasonality is found.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"`y` must be one-dimensional, got {y.ndim} dimensions.")
    if max_season_length is not None and max_season_length < 2:
        raise ValueError(
            f"`max_season_length` must be at least 2, got {max_season_length}."
        )
    y = _longest_finite_run(y)
    n = y.size
    if n < 4:
        return 1
    t = np.arange(n, dtype=np.float64)
    x = y - np.polyval(np.polyfit(t, y, deg=1), t)
    scale = x.std()
    if scale <= 1e-12 * max(1.0, np.abs(y).max()):
        return 1
    x = x / scale

    n_freq = 500
    threshold = 10.0
    freq, spec = _ar_spectrum(x, n_freq)
    if max_season_length is not None:
        in_band = freq >= 1.0 / max_season_length
        freq, spec = freq[in_band], spec[in_band]
        if spec.size == 0 or spec.max() <= threshold:
            return 1
        return int(np.floor(1.0 / freq[np.argmax(spec)] + 0.5))
    if spec.max() <= threshold:
        return 1
    peak = int(np.argmax(spec))
    if peak > 0:
        return int(np.floor(1.0 / freq[peak] + 0.5))
    # The peak is at frequency 0: use the next local maximum, as R does.
    rising = np.flatnonzero(np.diff(spec) > 0)
    if rising.size == 0:
        return 1
    next_peak = rising[0] + 1 + int(np.argmax(spec[rising[0] + 1 :]))
    if next_peak >= n_freq - 1:
        return 1
    return int(np.floor(1.0 / freq[next_peak] + 0.5))


def _repeat_val_seas(season_vals: np.ndarray, h: int) -> np.ndarray:
    repeats = math.ceil(h / season_vals.size)
    return np.tile(season_vals, repeats)[:h]


def _ensure_float(x: np.ndarray) -> np.ndarray:
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float32)
    return x


def _seasonal_naive(
    y: np.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
    season_length: int,  # season length
) -> Dict[str, np.ndarray]:
    y = _ensure_float(y)
    n = y.size
    season_vals = np.full(season_length, np.nan, dtype=y.dtype)
    n_available = min(season_length, n)
    if n < season_length:
        # Partial season: align observations to their seasonal positions
        # so that the forecast cycles correctly. 
        warnings.warn(
            f"Historical data ({n}) is shorter than season_length "
            f"({season_length}). Forecasts for positions without "
            f"observations will be filled with NaN."
        )
        start_idx = season_length - n_available
        season_vals[start_idx:] = y[-n_available:]
    else:
        season_vals[:n_available] = y[-n_available:]
    out = _repeat_val_seas(season_vals=season_vals, h=h)
    fcst = {"mean": out}
    if fitted:
        fitted_vals = np.empty_like(y)
        fitted_vals[:season_length] = np.nan
        if n > season_length:
            fitted_vals[season_length:] = y[: n - season_length]
        fcst["fitted"] = fitted_vals
    return fcst


def _repeat_val(val: float, h: int) -> np.ndarray:
    return np.full(h, val)


def _naive(
    y: np.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
) -> Dict[str, np.ndarray]:
    fcst = {"mean": _repeat_val(val=y[-1], h=h)}
    if fitted:
        fitted_vals = np.full_like(y, np.nan)
        fitted_vals[1:] = np.roll(y, 1)[1:]
        fcst["fitted"] = fitted_vals
    return fcst


class Distribution(str, Enum):
    NORMAL = "normal"
    LAPLACE = "laplace"
    T = "t"
    SKEW_NORMAL = "skew-normal"
    GED = "ged"

    def __str__(self):
        return self.value


class ArimaMethod(str, Enum):
    CSS = "CSS"
    ML = "ML"
    CSS_ML = "CSS-ML"


_VALID_DISTRIBUTIONS = tuple(Distribution)


# Functions used for calculating prediction intervals
def _quantiles(level, distribution="normal", dist_params=None):
    from .distributions import frozen_error_distribution
    p = 0.5 + np.asarray(level) / 200
    return frozen_error_distribution(1.0, distribution, dist_params).ppf(p)


def _calculate_intervals(out, level, h, sigmah, distribution="normal", dist_params=None):
    z = _quantiles(np.asarray(level), distribution=distribution, dist_params=dist_params)
    zz = np.repeat(z, h)
    zz = zz.reshape(z.shape[0], h)
    lower = out["mean"] - zz * sigmah
    upper = out["mean"] + zz * sigmah
    pred_int = {
        **{f"lo-{lv}": lower[i] for i, lv in enumerate(level)},
        **{f"hi-{lv}": upper[i] for i, lv in enumerate(level)},
    }
    return pred_int


def _calculate_sigma(residuals, n):
    if n > 0:
        sigma = np.nansum(residuals**2)
        sigma = sigma / n
        sigma = np.sqrt(sigma)
    else:
        sigma = 0
    return sigma


def _conformal_rank(n: int, q: float, upper: bool) -> int:
    r"""1-based rank of the finite-sample conformal quantile among `n` scores.

    Split conformal prediction takes the `ceil((n + 1) * q)`-th smallest score
    for an upper bound and the `floor((n + 1) * q)`-th for a lower bound, so
    each bound covers at least its nominal probability when the scores are
    exchangeable. The tail is explicit rather than inferred from `q`: an
    absolute-score margin is an upper bound even when `q < 0.5`. The rank is
    clipped to `[1, n]`: beyond that only an infinite bound would reach the
    nominal level.
    """
    # rounding guards against `(n + 1) * q` landing just off an integer
    pos = round((n + 1) * q, 8)
    k = math.ceil(pos) if upper else math.floor(pos)
    return min(max(k, 1), n)


def _conformal_quantile(scores, q, upper, axis: int = 0) -> np.ndarray:
    r"""Finite-sample conformal quantile(s) of `scores` along `axis`.

    Returns the exact order statistic of `_conformal_rank` rather than an
    interpolated quantile: linear interpolation lands between two scores and
    under-covers the lower tail. `upper` is a bool, or one bool per level in
    `q`. Like `np.quantile`, a sequence of levels adds a leading axis to the
    result.
    """
    scores = np.asarray(scores)
    n = scores.shape[axis]
    if n == 0:
        raise ValueError("Cannot compute a conformal quantile of zero scores")
    qs = np.atleast_1d(q)
    uppers = np.broadcast_to(np.asarray(upper, dtype=bool), qs.shape)
    ranks = [_conformal_rank(n, float(qi), bool(ui)) - 1 for qi, ui in zip(qs, uppers)]
    out = np.moveaxis(np.take(np.sort(scores, axis=axis), ranks, axis=axis), axis, 0)
    return out[0] if np.ndim(q) == 0 else out


class ConformalIntervals:
    """Class for storing conformal intervals metadata information.

    Args:
        n_windows (int, optional): Number of windows for conformal intervals. Defaults to 2.
        h (int, optional): Forecasting horizon. Defaults to 1.
        method (str, optional): Method for conformal intervals. Defaults to "conformal_distribution". Must be one of ["conformal_distribution", "conformal_error"].
    """

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        if h < 1:
            raise ValueError("h >= 1 to compute conformal intervals")
        allowed_methods = ["conformal_distribution", "conformal_error"]
        # Keep in sync with _get_conformal_method's available_methods dict in models.py
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method
