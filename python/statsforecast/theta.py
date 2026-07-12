__all__ = ["forecast_theta", "auto_theta", "forward_theta", "simulate_theta"]


import math

import numpy as np
import warnings
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from ._lib import theta as _theta
from .arima import is_constant
from .distributions import (
    switch_distribution,
    dist_init_params,
    extract_dist_params,
    aic_bic_aicc,
    error_params_from_model,
)
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
    distribution: str = "normal",
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

    if distribution == "normal":
        # Normal path: UNCHANGED — optimize MSE-based objective
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
    else:
        # Non-normal path: one-pass likelihood-based optimization
        struct_x0 = np.array(
            [par[k] for k, v in optimize_params.items() if v], dtype=np.float64
        )
        n_struct = len(struct_x0)
        var_init = max(float(np.nanvar(y)), 1e-10)
        n_dist, dist_init = dist_init_params(distribution, var_init)
        # structural bounds (same as optimize_theta_target_fn)
        lb = {"initial_smoothed": -1e10, "alpha": 0.1, "theta": 1.0}
        ub = {"initial_smoothed": 1e10, "alpha": 0.99, "theta": 1e10}
        lower = [lb[k] for k, v in optimize_params.items() if v]
        upper = [ub[k] for k, v in optimize_params.items() if v]
        x0_ext = np.concatenate([struct_x0, dist_init])
        lower_ext = np.concatenate([lower, np.full(n_dist, -np.inf)])
        upper_ext = np.concatenate([upper, np.full(n_dist, np.inf)])
        opt_res = _theta.optimize_dist(
            x0_ext, lower_ext, upper_ext,
            par["initial_smoothed"], par["alpha"], par["theta"],
            optimize_params["initial_smoothed"], optimize_params["alpha"],
            optimize_params["theta"], y, model_type, nmse,
            switch_distribution(distribution, _theta),
        )
        fred = results(*opt_res)
        fit_par = fred.x[:n_struct] if n_dist > 0 else fred.x
        fit_par_dist = fred.x[-n_dist:] if n_dist > 0 else np.array([])
        jj = 0
        for key in ("initial_smoothed", "alpha", "theta"):
            if optimize_params[key]:
                par[key] = fit_par[jj]
                jj += 1

    amse, e, states, mse = _theta.pegels_resid(
        y,
        model_type,
        par["initial_smoothed"],
        par["alpha"],
        par["theta"],
        nmse,
    )

    out = dict(
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
        distribution=distribution,
    )

    if distribution != "normal":
        n_eff = len(y) - 3
        n_free = sum(optimize_params.values())
        n_dist_count = 1 if distribution == "laplace" else 2
        np_eff = n_free + n_dist_count
        if distribution == "laplace":
            neg2logL = 2 * n_eff * fred.fn + n_eff * (2 + np.log(4))
        else:
            neg2logL = 2 * n_eff * fred.fn
        aic, bic, aicc = aic_bic_aicc(neg2logL, np_eff, n_eff)
        loglik = -0.5 * neg2logL
        dist_extra = extract_dist_params(distribution, fit_par_dist, residuals=e[3:])
        out["aic"] = aic
        out["bic"] = bic
        out["aicc"] = aicc
        out["loglik"] = loglik
        out.update(dist_extra)

    return out


def compute_pi_samples(
    n,
    h,
    states,
    sigma,
    alpha,
    theta,
    mean_y,
    seed=0,
    n_samples=200,
    error_distribution="normal",
    error_params=None,
    residuals=None,
):
    """
    Compute prediction interval samples for Theta model.

    Parameters
    ----------
    error_distribution : str, default='normal'
        Distribution for error terms. Options: 'normal', 't', 'bootstrap',
        'laplace', 'skew-normal', 'ged'.
    error_params : dict, optional
        Distribution-specific parameters.
    residuals : np.ndarray, optional
        Residuals for bootstrap sampling.
    """
    from statsforecast.simulation import sample_errors

    samples = np.full((h, n_samples), fill_value=np.nan, dtype=np.float32)
    # states: level, meany, An, Bn, mu
    smoothed, _, A, B, _ = states[-1]

    rng = np.random.default_rng(seed)

    for i in range(n, n + h):
        samples[i - n] = smoothed + (1 - 1 / theta) * (
            A * ((1 - alpha) ** i) + B * (1 - (1 - alpha) ** (i + 1)) / alpha
        )
        # Sample errors from specified distribution
        errors = sample_errors(
            size=n_samples,
            sigma=sigma,
            distribution=error_distribution,
            params=error_params,
            residuals=residuals,
            rng=rng,
        )
        samples[i - n] += errors
        smoothed = alpha * samples[i - n] + (1 - alpha) * smoothed
        mean_y = (i * mean_y + samples[i - n]) / (i + 1)
        B = ((i - 1) * B + 6 * (samples[i - n] - mean_y) / (i + 1)) / (i + 2)
        A = mean_y - B * (i + 2) / 2
    return samples


def simulate_theta(
    model,
    h,
    n_paths,
    seed=None,
    error_distribution="normal",
    error_params=None,
):
    """
    Simulate future paths from a fitted Theta model.

    Parameters
    ----------
    model : dict
        Fitted Theta model dictionary.
    h : int
        Forecast horizon.
    n_paths : int
        Number of simulation paths to generate.
    seed : int, optional
        Random seed for reproducibility.
    error_distribution : str, default='normal'
        Distribution for error terms. Options: 'normal', 't', 'bootstrap',
        'laplace', 'skew-normal', 'ged'.
    error_params : dict, optional
        Distribution-specific parameters. E.g., {'df': 5} for t-distribution.

    Returns
    -------
    np.ndarray
        Simulated paths of shape (n_paths, h).
    """
    # Set up random generator
    if seed is not None:
        np.random.seed(seed)

    residuals_tail = model["residuals"][3:]
    if len(residuals_tail) < 2:
        warnings.warn(
            "Too few residuals after burn-in for sigma estimate; using all residuals",
            stacklevel=2,
        )
        sigma = np.std(model["residuals"], ddof=1)
    else:
        sigma = np.std(residuals_tail, ddof=1)
    residuals = residuals_tail if len(residuals_tail) >= 2 else model["residuals"]

    samples = compute_pi_samples(
        n=model["n"],
        h=h,
        states=model["states"],
        sigma=sigma,
        alpha=model["par"]["alpha"],
        theta=model["par"]["theta"],
        mean_y=model["mean_y"],
        seed=seed if seed is not None else 0,
        n_samples=n_paths,
        error_distribution=error_distribution,
        error_params=error_params,
        residuals=residuals,
    )

    res = samples.T

    if model.get("decompose", False):
        seas_forecast = _repeat_val_seas(model["seas_forecast"]["mean"], h=h)
        if model["decomposition_type"] == "multiplicative":
            res = res * seas_forecast
        else:
            res = res + seas_forecast
    return res



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
        residuals_tail = obj["residuals"][3:]
        if len(residuals_tail) < 2:
            warnings.warn(
                "Too few residuals after burn-in for sigma estimate; using all residuals",
                stacklevel=2,
            )
            sigma = np.std(obj["residuals"], ddof=1)
        else:
            sigma = np.std(residuals_tail, ddof=1)
        mean_y = obj["mean_y"]
        dist = obj.get("distribution", "normal")
        samples = compute_pi_samples(
            n=n,
            h=h,
            states=states,
            sigma=sigma,
            alpha=alpha,
            theta=theta,
            mean_y=mean_y,
            error_distribution=dist,
            error_params=error_params_from_model(obj),
            residuals=residuals_tail,
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
    distribution="normal",
):
    # converting params to floats
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
            distribution=distribution,
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

    ic_key = "mse" if distribution == "normal" else "aic"
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
            distribution=distribution,
        )
        fit_ic = fit[ic_key]
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
