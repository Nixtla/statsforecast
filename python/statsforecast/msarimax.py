import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from os import cpu_count
from typing import Iterable, Optional, Union

import numpy as np
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class MSARIMAXOrder:
    lags: list[int]
    ar_orders: list[int]
    i_orders: list[int]
    ma_orders: list[int]
    ar_lags: list[int]
    ma_lags: list[int]
    delta: np.ndarray


def normalize_msarimax_orders(
    lags: Iterable[int] = (1,),
    ar_order: Union[int, Iterable[int]] = 0,
    i_order: Union[int, Iterable[int]] = 1,
    ma_order: Union[int, Iterable[int]] = 1,
) -> MSARIMAXOrder:
    lags, ar_orders, i_orders, ma_orders = _normalise_spec(
        lags, ar_order, i_order, ma_order
    )
    ar_lags = _polynomial_lags(lags, ar_orders)
    ma_lags = _polynomial_lags(lags, ma_orders)
    delta = _difference_polynomial(lags, i_orders)
    return MSARIMAXOrder(lags, ar_orders, i_orders, ma_orders, ar_lags, ma_lags, delta)


def msarimax(
    y: np.ndarray,
    lags: Iterable[int] = (1,),
    ar_order: Union[int, Iterable[int]] = 0,
    i_order: Union[int, Iterable[int]] = 1,
    ma_order: Union[int, Iterable[int]] = 1,
    xreg: Optional[np.ndarray] = None,
    include_constant: bool = False,
    method: str = "lbfgs",
    maxiter: int = 200,
):
    order = normalize_msarimax_orders(lags, ar_order, i_order, ma_order)
    y = np.asarray(y, dtype=np.float64)
    z = difference_with_polynomial(y, order.delta)
    if z.size == 0:
        raise ValueError("not enough data to apply the requested differencing")
    exog = None
    if xreg is not None:
        xreg = np.asarray(xreg, dtype=np.float64)
        if xreg.shape[0] != y.shape[0]:
            raise ValueError("lengths of `y` and `xreg` do not match")
        exog = difference_matrix_with_polynomial(xreg, order.delta)
    trend = "c" if include_constant else "n"
    model = SARIMAX(
        z,
        exog=exog,
        order=(order.ar_lags, 0, order.ma_lags),
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(method=method, maxiter=maxiter, disp=False)
    fitted_z = np.asarray(fit.fittedvalues, dtype=np.float64)
    fitted = invert_fitted_values(y, fitted_z, order.delta)
    resid = y - fitted
    resid[: len(y) - len(fitted_z)] = np.nan
    return {
        "fit": fit,
        "order": order,
        "x": y,
        "xreg": xreg,
        "fitted": fitted,
        "residuals": resid,
        "sigma2": float(np.nanmean((z - fitted_z) ** 2)),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "aicc": _aicc(float(fit.aic), fit.nobs, len(fit.params)),
    }


def forecast_msarimax(model, h: int, xreg: Optional[np.ndarray] = None, level=None):
    order = model["order"]
    future_exog = None
    if xreg is not None:
        xreg = np.asarray(xreg, dtype=np.float64)
        if xreg.shape[0] != h:
            raise ValueError("future xreg must have the same number of rows as `h`")
        if model["xreg"] is None:
            raise ValueError("future xreg supplied, but the model was fit without xreg")
        full_xreg = np.vstack([model["xreg"], xreg])
        future_exog = difference_matrix_with_polynomial(full_xreg, order.delta)[-h:]
    pred = model["fit"].get_forecast(steps=h, exog=future_exog)
    mean_z = np.asarray(pred.predicted_mean, dtype=np.float64)
    mean = invert_forecast_values(model["x"], mean_z, order.delta)
    out = {"mean": mean}
    if level is not None:
        se = np.asarray(pred.se_mean, dtype=np.float64)
        level = sorted(level)
        lower = {}
        upper = {}
        for lv in level:
            q = norm.ppf(0.5 + lv / 200)
            lower[f"{lv}%"] = invert_forecast_values(
                model["x"], mean_z - q * se, order.delta
            )
            upper[f"{lv}%"] = invert_forecast_values(
                model["x"], mean_z + q * se, order.delta
            )
        out["lower"] = lower
        out["upper"] = upper
    return out


def fitted_msarimax(model):
    return model["fitted"]


def auto_msarimax(
    y: np.ndarray,
    lags: Iterable[int] = (1,),
    max_ar_order: Union[int, Iterable[int]] = (3, 3),
    max_i_order: Union[int, Iterable[int]] = (2, 1),
    max_ma_order: Union[int, Iterable[int]] = (3, 3),
    ic: str = "aicc",
    xreg: Optional[np.ndarray] = None,
    include_constant: Union[bool, Iterable[bool]] = (False, True),
    method: str = "lbfgs",
    maxiter: int = 200,
    n_jobs: int = 1,
):
    lags, max_ar, max_i, max_ma = _normalise_spec(
        lags, max_ar_order, max_i_order, max_ma_order
    )
    constants = (
        [bool(include_constant)]
        if isinstance(include_constant, (bool, np.bool_))
        else [bool(v) for v in include_constant]
    )
    candidates = []
    for ar_orders in product(*[range(v + 1) for v in max_ar]):
        for i_orders in product(*[range(v + 1) for v in max_i]):
            for ma_orders in product(*[range(v + 1) for v in max_ma]):
                if not any(ar_orders + i_orders + ma_orders):
                    continue
                for constant in constants:
                    candidates.append(
                        (
                            list(ar_orders),
                            list(i_orders),
                            list(ma_orders),
                            bool(constant),
                        )
                    )
    tried = len(candidates)
    if n_jobs == 1:
        results = [
            _fit_auto_msarimax_candidate(
                y, lags, xreg, method, maxiter, ic, candidate
            )
            for candidate in candidates
        ]
        best = None
        best_value = np.inf
        failed = 0
        for fit, value, constant in results:
            if fit is None:
                failed += 1
                continue
            if np.isfinite(value) and value < best_value:
                best = fit
                best_value = value
                best["include_constant"] = constant
    else:
        max_workers = _effective_n_jobs(n_jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    _score_auto_msarimax_candidate_from_args,
                    (
                        (y, lags, xreg, method, maxiter, ic, candidate)
                        for candidate in candidates
                    ),
                )
            )
        best_candidate = None
        best_value = np.inf
        failed = 0
        for value, candidate in results:
            if not np.isfinite(value):
                failed += 1
                continue
            if value < best_value:
                best_value = value
                best_candidate = candidate
        best = None
        if best_candidate is not None:
            ar_orders, i_orders, ma_orders, constant = best_candidate
            best = msarimax(
                y=y,
                lags=lags,
                ar_order=ar_orders,
                i_order=i_orders,
                ma_order=ma_orders,
                xreg=xreg,
                include_constant=constant,
                method=method,
                maxiter=maxiter,
            )
            best["include_constant"] = constant
    if best is None:
        raise ValueError("no MSARIMAX model could be estimated")
    best["selection"] = {
        "tried": tried,
        "failed": failed,
        "ic": ic,
        "ic_value": best_value,
    }
    return best


def _score_auto_msarimax_candidate_from_args(args):
    y, lags, xreg, method, maxiter, ic, candidate = args
    fit, value, _ = _fit_auto_msarimax_candidate(
        y, lags, xreg, method, maxiter, ic, candidate
    )
    return (value if fit is not None else np.inf), candidate


def _fit_auto_msarimax_candidate(y, lags, xreg, method, maxiter, ic, candidate):
    ar_orders, i_orders, ma_orders, constant = candidate
    try:
        fit = msarimax(
            y=y,
            lags=lags,
            ar_order=ar_orders,
            i_order=i_orders,
            ma_order=ma_orders,
            xreg=xreg,
            include_constant=constant,
            method=method,
            maxiter=maxiter,
        )
    except Exception:
        return None, np.inf, constant
    return fit, _ic_value(fit, ic), constant


def _effective_n_jobs(n_jobs: int) -> int:
    if n_jobs == 0:
        raise ValueError("n_jobs must be different from 0")
    if n_jobs < 0:
        return max((cpu_count() or 1) + 1 + n_jobs, 1)
    return n_jobs


def difference_with_polynomial(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    max_lag = len(delta) - 1
    if max_lag == 0:
        return y.copy()
    out = np.convolve(y, delta, mode="valid")
    return out


def difference_matrix_with_polynomial(x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack(
        [difference_with_polynomial(x[:, i], delta) for i in range(x.shape[1])]
    )


def invert_fitted_values(
    y: np.ndarray, fitted_z: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    max_lag = len(delta) - 1
    fitted = np.full_like(y, np.nan, dtype=np.float64)
    if max_lag == 0:
        fitted[:] = fitted_z
        return fitted
    for i, zhat in enumerate(fitted_z, start=max_lag):
        previous = y[i - np.arange(1, max_lag + 1)]
        fitted[i] = zhat - np.dot(delta[1:], previous)
    return fitted


def invert_forecast_values(
    y: np.ndarray, forecast_z: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    max_lag = len(delta) - 1
    if max_lag == 0:
        return forecast_z.copy()
    history = list(np.asarray(y, dtype=np.float64))
    out = []
    for zhat in forecast_z:
        previous = np.array(history[-1 : -max_lag - 1 : -1], dtype=np.float64)
        yhat = float(zhat - np.dot(delta[1:], previous))
        history.append(yhat)
        out.append(yhat)
    return np.asarray(out, dtype=np.float64)


def _normalise_spec(lags, ar_order, i_order, ma_order):
    lags = _as_int_list(lags)
    if not lags or lags[0] != 1:
        lags = [1] + lags
    orders = [_as_int_list(v) for v in (ar_order, i_order, ma_order)]
    n = max(len(lags), *(len(v) for v in orders))
    lags = lags + [0] * (n - len(lags))
    orders = [v + [0] * (n - len(v)) for v in orders]
    keep = [i for i, lag in enumerate(lags) if lag > 0]
    lags = [lags[i] for i in keep]
    orders = [[v[i] for i in keep] for v in orders]
    for values in (lags, *orders):
        if any(v < 0 for v in values):
            raise ValueError("lags and orders must be non-negative")
    return lags, *orders


def _as_int_list(value) -> list[int]:
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


def _polynomial_lags(lags: list[int], orders: list[int]) -> list[int]:
    poly = np.array([1.0])
    for lag, order in zip(lags, orders):
        factor = np.zeros(order * lag + 1)
        factor[0] = 1.0
        for i in range(order):
            factor[(i + 1) * lag] = 1.0
        poly = np.convolve(poly, factor)
    return [i for i, value in enumerate(poly[1:], start=1) if abs(value) > 1e-12]


def _difference_polynomial(lags: list[int], orders: list[int]) -> np.ndarray:
    poly = np.array([1.0])
    for lag, order in zip(lags, orders):
        factor = np.zeros(order * lag + 1)
        for k in range(order + 1):
            factor[k * lag] = (-1) ** k * math.comb(order, k)
        poly = np.convolve(poly, factor)
    return _trim(poly)


def _trim(poly: np.ndarray) -> np.ndarray:
    last = len(poly) - 1
    while last > 0 and abs(poly[last]) <= 1e-12:
        last -= 1
    return poly[: last + 1]


def _aicc(aic: float, nobs: int, n_params: int) -> float:
    denom = nobs - n_params - 1
    if denom <= 0:
        return np.inf
    return aic + 2 * n_params * (n_params + 1) / denom


def _ic_value(model, ic: str) -> float:
    name = ic.lower()
    if name == "aic":
        return model["aic"]
    if name == "bic":
        return model["bic"]
    if name in ("aicc", "aic_c"):
        return model["aicc"]
    raise ValueError("ic must be one of 'aic', 'aicc', or 'bic'")
