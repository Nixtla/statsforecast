__all__ = ['forecast_theta', 'auto_theta', 'forward_theta']


import math

import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from ._lib import theta as _theta
from .arima import is_constant
from .utils import _repeat_val_seas, _seasonal_naive, results


def initparamtheta(
    initial_smoothed: float,
    alpha: float,
    theta: float,
    y: np.ndarray,
    modeltype: _theta.ModelType,
):
    if modeltype in [_theta.ModelType.STM, _theta.ModelType.DSTM]:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        theta = 2.0  # no optimize
        optimize_theta = False
    else:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        if math.isnan(theta):
            theta = 2.0
            optimize_theta = True
        else:
            optimize_theta = False
    return {
        "initial_smoothed": initial_smoothed,
        "optimize_initial_smoothed": optimize_level,
        "alpha": alpha,
        "optimize_alpha": optimize_alpha,
        "theta": theta,
        "optimize_theta": optimize_theta,
    }


def switch_theta(model: str) -> _theta.ModelType:
    if model == "STM":
        return _theta.ModelType.STM
    if model == "OTM":
        return _theta.ModelType.OTM
    if model == "DSTM":
        return _theta.ModelType.DSTM
    if model == "DOTM":
        return _theta.ModelType.DOTM
    raise ValueError(f"Invalid model type: {model}.")


def optimize_theta_target_fn(
    init_par: dict[str, float],
    optimize_params: dict[str, bool],
    y: np.ndarray,
    modeltype: _theta.ModelType,
    nmse: int,
):
    x0 = []
    lower = []
    upper = []
    lower_bounds = {
        "initial_smoothed": -1e10,
        "alpha": 0.1,
        "theta": 1.0,
    }
    upper_bounds = {
        "initial_smoothed": 1e10,
        "alpha": 0.99,
        "theta": 1e10,
    }
    for param, optim in optimize_params.items():
        if optim:
            x0.append(init_par[param])
            lower.append(lower_bounds[param])
            upper.append(upper_bounds[param])
    if not x0:
        return
    x0 = np.array(x0)
    lower = np.array(lower)
    upper = np.array(upper)

    init_level = init_par["initial_smoothed"]
    init_alpha = init_par["alpha"]
    init_theta = init_par["theta"]

    opt_level = optimize_params["initial_smoothed"]
    opt_alpha = optimize_params["alpha"]
    opt_theta = optimize_params["theta"]

    opt_res = _theta.optimize(
        x0,
        lower,
        upper,
        init_level,
        init_alpha,
        init_theta,
        opt_level,
        opt_alpha,
        opt_theta,
        y,
        modeltype,
        nmse,
    )
    return results(*opt_res)


def thetamodel(
    y: np.ndarray,
    m: int,
    modeltype: str,
    initial_smoothed: float,
    alpha: float,
    theta: float,
    nmse: int,
):
    y = y.astype(np.float64, copy=False)
    model_type = switch_theta(modeltype)
    # initial parameters
    par = initparamtheta(
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
        y=y,
        modeltype=model_type,
    )
    optimize_params = {
        key.replace("optimize_", ""): val for key, val in par.items() if "optim" in key
    }
    par = {key: val for key, val in par.items() if "optim" not in key}
    # parameter optimization
    fred = optimize_theta_target_fn(
        init_par=par,
        optimize_params=optimize_params,
        y=y,
        modeltype=model_type,
        nmse=nmse,
    )
    if fred is not None:
        fit_par = fred.x
    j = 0
    if optimize_params["initial_smoothed"]:
        par["initial_smoothed"] = fit_par[j]
        j += 1
    if optimize_params["alpha"]:
        par["alpha"] = fit_par[j]
        j += 1
    if optimize_params["theta"]:
        par["theta"] = fit_par[j]
        j += 1

    amse, e, states, mse = _theta.pegels_resid(
        y,
        model_type,
        par["initial_smoothed"],
        par["alpha"],
        par["theta"],
        nmse,
    )

    return dict(
        mse=mse,
        amse=amse,
        fit=fred,
        residuals=e,
        m=m,
        states=states,
        par=par,
        n=len(y),
        modeltype=modeltype,
        mean_y=np.mean(y),
    )


def compute_pi_samples(
    n, h, states, sigma, alpha, theta, mean_y, seed=0, n_samples=200
):
    samples = np.full((h, n_samples), fill_value=np.nan, dtype=np.float32)
    # states: level, meany, An, Bn, mu
    smoothed, _, A, B, _ = states[-1]
    np.random.seed(seed)
    for i in range(n, n + h):
        samples[i - n] = smoothed + (1 - 1 / theta) * (
            A * ((1 - alpha) ** i) + B * (1 - (1 - alpha) ** (i + 1)) / alpha
        )
        samples[i - n] += np.random.normal(scale=sigma, size=n_samples)
        smoothed = alpha * samples[i - n] + (1 - alpha) * smoothed
        mean_y = (i * mean_y + samples[i - n]) / (i + 1)
        B = ((i - 1) * B + 6 * (samples[i - n] - mean_y) / (i + 1)) / (i + 2)
        A = mean_y - B * (i + 2) / 2
    return samples


def forecast_theta(obj, h, level=None):
    forecast = np.empty(h)
    n = obj["n"]
    states = obj["states"]
    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]
    _theta.forecast(
        states,
        n,
        switch_theta(obj["modeltype"]),
        forecast,
        alpha,
        theta,
    )
    res = {"mean": forecast}

    if level is not None:
        sigma = np.std(obj["residuals"][3:], ddof=1)
        mean_y = obj["mean_y"]
        samples = compute_pi_samples(
            n=n,
            h=h,
            states=states,
            sigma=sigma,
            alpha=alpha,
            theta=theta,
            mean_y=mean_y,
        )
        for lv in level:
            min_q = (100 - lv) / 200
            max_q = min_q + lv / 100
            res[f"lo-{lv}"] = np.quantile(samples, min_q, axis=1)
            res[f"hi-{lv}"] = np.quantile(samples, max_q, axis=1)

    if obj.get("decompose", False):
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        for key in res:
            if obj["decomposition_type"] == "multiplicative":
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast
    return res


def auto_theta(
    y,
    m,
    model=None,
    initial_smoothed=None,
    alpha=None,
    theta=None,
    nmse=3,
    decomposition_type="multiplicative",
):
    # converting params to floats
    # to improve numba compilation
    if initial_smoothed is None:
        initial_smoothed = np.nan
    if alpha is None:
        alpha = np.nan
    if theta is None:
        theta = np.nan
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    # constan values
    if is_constant(y):
        thetamodel(
            y=y,
            m=m,
            modeltype="STM",
            nmse=nmse,
            initial_smoothed=np.mean(y) / 2,
            alpha=0.5,
            theta=2.0,
        )
    # seasonal decomposition if needed
    decompose = False
    # seasonal test
    if m >= 4 and len(y) >= 2 * m:
        r = acf(y, nlags=m, fft=False)[1:]
        stat = np.sqrt((1 + 2 * np.sum(r[:-1] ** 2)) / len(y))
        decompose = np.abs(r[-1]) / stat > norm.ppf(0.95)

    data_positive = min(y) > 0
    if decompose:
        # change decomposition type if data is not positive
        if decomposition_type == "multiplicative" and not data_positive:
            decomposition_type = "additive"
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m).seasonal
        if decomposition_type == "multiplicative" and any(y_decompose < 0.01):
            decomposition_type = "additive"
            y_decompose = seasonal_decompose(y, model="additive", period=m).seasonal
        if decomposition_type == "additive":
            y = y - y_decompose
        else:
            y = y / y_decompose
        seas_forecast = _seasonal_naive(
            y=y_decompose, h=m, season_length=m, fitted=False
        )

    # validate model
    if model not in [None, "STM", "OTM", "DSTM", "DOTM"]:
        raise ValueError(f"Invalid model type: {model}.")

    n = len(y)
    npars = 3
    # non-optimized tiny datasets
    if n <= npars:
        raise NotImplementedError("tiny datasets")
    if model is None:
        modeltype = ["STM", "OTM", "DSTM", "DOTM"]
    else:
        modeltype = [model]

    best_ic = np.inf
    for mtype in modeltype:
        fit = thetamodel(
            y=y,
            m=m,
            modeltype=mtype,
            nmse=nmse,
            initial_smoothed=initial_smoothed,
            alpha=alpha,
            theta=theta,
        )
        fit_ic = fit["mse"]
        if not np.isnan(fit_ic):
            if fit_ic < best_ic:
                model = fit
                best_ic = fit_ic
    if np.isinf(best_ic):
        raise Exception("no model able to be fitted")

    if decompose:
        if decomposition_type == "multiplicative":
            model["residuals"] = model["residuals"] * y_decompose
        else:
            model["residuals"] = model["residuals"] + y_decompose
        model["decompose"] = decompose
        model["decomposition_type"] = decomposition_type
        model["seas_forecast"] = dict(seas_forecast)
    return model


def forward_theta(fitted_model, y):
    m = fitted_model["m"]
    model = fitted_model["modeltype"]
    initial_smoothed = fitted_model["par"]["initial_smoothed"]
    alpha = fitted_model["par"]["alpha"]
    theta = fitted_model["par"]["theta"]
    return auto_theta(
        y=y,
        m=m,
        model=model,
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
    )
