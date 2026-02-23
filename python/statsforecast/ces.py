__all__ = ["auto_ces", "simulate_ces"]


import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from ._lib import ces as _ces
from .utils import results

# Global variables
NONE = 0
SIMPLE = 1
PARTIAL = 2
FULL = 3
TOL = 1.0e-10
HUGEN = 1.0e10
NA = -99999.0
smalno = np.finfo(float).eps


def initstate(y, m, seasontype):
    n = len(y)
    components = 2 + (seasontype == "P") + 2 * (seasontype == "F")
    lags = 1 if seasontype == "N" else m
    states = np.zeros((lags, components), dtype=np.float64)
    if seasontype == "N":
        idx = min(max(10, m), n)
        mean_ = np.mean(y[:idx])
        states[0, 0] = mean_
        states[0, 1] = mean_ / 1.1
    elif seasontype == "S":
        states[:lags, 0] = y[:lags]
        states[:lags, 1] = y[:lags] / 1.1
    elif seasontype == "P":
        states[:lags, 0] = np.mean(y[:lags])
        states[:lags, 1] = states[:lags, 0] / 1.1
        states[:lags, 2] = seasonal_decompose(y, period=lags).seasonal[:lags]
    elif seasontype == "F":
        states[:lags, 0] = np.mean(y[:lags])
        states[:lags, 1] = states[:lags, 0] / 1.1
        states[:lags, 2] = seasonal_decompose(y, period=lags).seasonal[:lags]
        states[:lags, 3] = states[:lags, 2] / 1.1
    else:
        raise Exception(f"Unkwon seasontype: {seasontype}")

    return states


def switch_ces(x: str):
    return _ces.switch_ces(x)


def initparamces(
    alpha_0: float, alpha_1: float, beta_0: float, beta_1: float, seasontype: str
):
    if np.isnan(alpha_0):
        alpha_0 = 1.3
        optimize_alpha_0 = 1
    else:
        optimize_alpha_0 = 0
    if np.isnan(alpha_1):
        alpha_1 = 1.0
        optimize_alpha_1 = 1
    else:
        optimize_alpha_1 = 0
    if seasontype == "P":
        if np.isnan(beta_0):
            beta_0 = 0.1
            optimize_beta_0 = 1
        else:
            optimize_beta_0 = 0
        beta_1 = np.nan  # no optimize
        optimize_beta_1 = 0
    elif seasontype == "F":
        if np.isnan(beta_0):
            beta_0 = 1.3
            optimize_beta_0 = 1
        else:
            optimize_beta_0 = 0
        if np.isnan(beta_1):
            beta_1 = 1.0
            optimize_beta_1 = 1
        else:
            optimize_beta_1 = 0
    else:
        # no optimize
        optimize_beta_0 = 0
        optimize_beta_1 = 0
        beta_0 = np.nan
        beta_1 = np.nan
    return {
        "alpha_0": alpha_0,
        "optimize_alpha_0": optimize_alpha_0,
        "alpha_1": alpha_1,
        "optimize_alpha_1": optimize_alpha_1,
        "beta_0": beta_0,
        "optimize_beta_0": optimize_beta_0,
        "beta_1": beta_1,
        "optimize_beta_1": optimize_beta_1,
    }


def cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1,
            e, amse, nmse, backfit):
    return _ces.cescalc(
        np.asarray(y, dtype=np.float64),
        states,
        m, season,
        float(alpha_0), float(alpha_1), float(beta_0), float(beta_1),
        e, amse, nmse, backfit,
    )


def pegelsresid_ces(
    y: np.ndarray,
    m: int,
    init_states: np.ndarray,
    n_components: int,
    seasontype: str,
    alpha_0: float,
    alpha_1: float,
    beta_0: float,
    beta_1: float,
    nmse: int,
):
    return _ces.pegelsresid(
        np.asarray(y, dtype=np.float64),
        m,
        np.asarray(init_states, dtype=np.float64),
        n_components,
        seasontype,
        float(alpha_0),
        float(alpha_1),
        float(beta_0),
        float(beta_1),
        nmse,
    )


def cesforecast(states, n, m, season, f, h, alpha_0, alpha_1, beta_0, beta_1):
    seasontype_map = {0: "N", 1: "S", 2: "P", 3: "F"}
    seasontype_str = seasontype_map[season]
    f_f64 = np.zeros(h, dtype=np.float64)
    _ces.forecast(
        np.asarray(states, dtype=np.float64),
        n, m, seasontype_str, f_f64, h,
        float(alpha_0), float(alpha_1), float(beta_0), float(beta_1),
    )
    f[:] = f_f64.astype(f.dtype)


def optimize_ces_target_fn(
    init_par, optimize_params, y, m, init_states, n_components, seasontype, nmse
):
    x0 = [init_par[key] for key, val in optimize_params.items() if val]
    x0 = np.array(x0, dtype=np.float64)
    if not len(x0):
        return

    init_alpha_0 = float(init_par["alpha_0"])
    init_alpha_1 = float(init_par["alpha_1"])
    init_beta_0 = float(init_par["beta_0"])
    init_beta_1 = float(init_par["beta_1"])

    opt_alpha_0 = bool(optimize_params["alpha_0"])
    opt_alpha_1 = bool(optimize_params["alpha_1"])
    opt_beta_0 = bool(optimize_params["beta_0"])
    opt_beta_1 = bool(optimize_params["beta_1"])

    lower = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    upper = np.array([1.8, 1.9, 1.5, 1.5], dtype=np.float64)

    opt_result = _ces.optimize(
        x0, lower, upper,
        init_alpha_0, init_alpha_1, init_beta_0, init_beta_1,
        opt_alpha_0, opt_alpha_1, opt_beta_0, opt_beta_1,
        np.asarray(y, dtype=np.float64), m,
        np.asarray(init_states, dtype=np.float64),
        n_components, seasontype, nmse,
    )
    # opt_result is (x, fn, nit, simplex) tuple from C++ Nelder-Mead
    return results(opt_result[0], opt_result[1], opt_result[2], opt_result[3])


def cesmodel(
    y: np.ndarray,
    m: int,
    seasontype: str,
    alpha_0: float,
    alpha_1: float,
    beta_0: float,
    beta_1: float,
    nmse: int,
):
    if seasontype == "N":
        m = 1
    # initial parameters
    par = initparamces(
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        beta_1=beta_1,
        beta_0=beta_0,
        seasontype=seasontype,
    )
    optimize_params = {
        key.replace("optimize_", ""): val for key, val in par.items() if "optim" in key
    }
    par = {key: val for key, val in par.items() if "optim" not in key}
    # initial states
    init_state = initstate(y, m, seasontype)
    n_components = init_state.shape[1]
    # parameter optimization
    fred = optimize_ces_target_fn(
        init_par=par,
        optimize_params=optimize_params,
        y=y,
        m=m,
        init_states=init_state,
        n_components=n_components,
        seasontype=seasontype,
        nmse=nmse,
    )
    if fred is not None:
        fit_par = fred.x
    j = 0
    if optimize_params["alpha_0"]:
        par["alpha_0"] = fit_par[j]
        j += 1
    if optimize_params["alpha_1"]:
        par["alpha_1"] = fit_par[j]
        j += 1
    if optimize_params["beta_0"]:
        par["beta_0"] = fit_par[j]
        j += 1
    if optimize_params["beta_1"]:
        par["beta_1"] = fit_par[j]
        j += 1

    amse, e, states, lik = pegelsresid_ces(
        y=y,
        m=m,
        init_states=init_state,
        n_components=n_components,
        seasontype=seasontype,
        nmse=nmse,
        **par,
    )
    np_ = n_components + 1
    ny = len(y)
    aic = lik + 2 * np_
    bic = lik + np.log(ny) * np_
    if ny - np_ - 1 != 0.0:
        aicc = aic + 2 * np_ * (np_ + 1) / (ny - np_ - 1)
    else:
        aicc = np.inf

    mse = amse[0]
    amse = np.mean(amse)

    fitted = y - e
    sigma2 = np.sum(e**2) / (ny - np_ - 1)

    return dict(
        loglik=-0.5 * lik,
        aic=aic,
        bic=bic,
        aicc=aicc,
        mse=mse,
        amse=amse,
        fit=fred,
        fitted=fitted,
        residuals=e,
        m=m,
        states=states,
        par=par,
        n=len(y),
        seasontype=seasontype,
        sigma2=sigma2,
    )


def pegelsfcast_C(h, obj, npaths=None, level=None, bootstrap=None):
    forecast = np.full(h, fill_value=np.nan)
    m = obj["m"]
    n = obj["n"]
    states = obj["states"]
    season = switch_ces(obj["seasontype"])
    cesforecast(
        states=states,
        n=n,
        m=m,
        season=season,
        h=h,
        f=forecast,
        **obj["par"],
    )
    return forecast


def _simulate_pred_intervals(model, h, level):
    np.random.seed(1)
    nsim = 5000
    y_path = np.zeros([nsim, h])

    season = switch_ces(model["seasontype"])
    for k in range(nsim):
        e = np.random.normal(0, np.sqrt(model["sigma2"]), model["states"].shape)
        states = model["states"]
        fcsts = np.zeros(h, dtype=np.float64)
        cesforecast(
            states=states + e,
            n=model["n"],
            m=model["m"],
            season=season,
            h=h,
            f=fcsts,
            **model["par"],
        )
        y_path[k,] = fcsts

    lower = np.quantile(y_path, 0.5 - np.array(level) / 200, axis=0)
    upper = np.quantile(y_path, 0.5 + np.array(level) / 200, axis=0)
    pi = {
        **{f"lo-{lv}": lower[i] for i, lv in enumerate(level)},
        **{f"hi-{lv}": upper[i] for i, lv in enumerate(level)},
    }

    return pi


def forecast_ces(obj, h, level=None):
    fcst = pegelsfcast_C(h, obj)
    out = {"mean": fcst}
    out["fitted"] = obj["fitted"]
    if level is not None:
        pi = _simulate_pred_intervals(model=obj, h=h, level=level)
        out = {**out, **pi}
    return out


def auto_ces(
    y,
    m,
    model="Z",
    alpha_0=None,
    alpha_1=None,
    beta_0=None,
    beta_1=None,
    opt_crit="lik",
    nmse=3,
    ic="aicc",
):
    # converting params to floats
    if alpha_0 is None:
        alpha_0 = np.nan
    if alpha_1 is None:
        alpha_1 = np.nan
    if beta_0 is None:
        beta_0 = np.nan
    if beta_1 is None:
        beta_1 = np.nan
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    # refit model not implement yet
    if model not in ["Z", "N", "S", "P", "F"]:
        raise ValueError("Invalid model type")

    seasontype = model
    if m < 1 or len(y) < 2 * m or m == 1:
        seasontype = "N"
    n = len(y)
    npars = 2
    if seasontype == "P":
        npars += 1
    if seasontype in ["F", "Z"]:
        npars += 2
    # ses for non-optimized tiny datasets
    if n <= npars:
        # we need HoltWintersZZ function
        raise NotImplementedError("tiny datasets")
    if seasontype == "Z":
        seasontype = ["N", "S", "P", "F"]
    best_ic = np.inf
    for stype in seasontype:
        fit = cesmodel(
            y=y,
            m=m,
            seasontype=stype,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            beta_0=beta_0,
            beta_1=beta_1,
            nmse=nmse,
        )
        fit_ic = fit[ic]
        if not np.isnan(fit_ic):
            if fit_ic < best_ic:
                model = fit
                best_ic = fit_ic
    if np.isinf(best_ic):
        raise Exception("no model able to be fitted")
    return model


def forward_ces(fitted_model, y):
    m = fitted_model["m"]
    model = fitted_model["seasontype"]
    alpha_0 = fitted_model["par"]["alpha_0"]
    alpha_1 = fitted_model["par"]["alpha_1"]
    beta_0 = fitted_model["par"]["beta_0"]
    beta_1 = fitted_model["par"]["beta_1"]
    return auto_ces(
        y=y,
        m=m,
        model=model,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        beta_0=beta_0,
        beta_1=beta_1,
    )


def simulate_ces(
    model,
    h,
    n_paths,
    seed=None,
    error_distribution="normal",
    error_params=None,
):
    """
    Simulate future paths from a fitted CES model.

    Parameters
    ----------
    model : dict
        Fitted CES model dictionary.
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
    from statsforecast.simulation import sample_errors

    # Set up random generator
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    y_path = np.zeros([n_paths, h])

    sigma = np.sqrt(model["sigma2"])
    states_shape = model["states"].shape

    # Get residuals for bootstrap if needed
    residuals = model.get("residuals", None)

    season = switch_ces(model["seasontype"])
    for k in range(n_paths):
        # Sample state noise from specified distribution
        e = sample_errors(
            size=states_shape,
            sigma=sigma,
            distribution=error_distribution,
            params=error_params,
            residuals=residuals,
            rng=rng,
        )
        states = model["states"]
        fcsts = np.zeros(h, dtype=np.float64)
        cesforecast(
            states=states + e,
            n=model["n"],
            m=model["m"],
            season=season,
            h=h,
            f=fcsts,
            **model["par"],
        )
        y_path[k,] = fcsts

    return y_path

