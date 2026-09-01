"""Single hub for error-distribution definitions and MLE-fitting helpers.

Re-exports the canonical Distribution enum and quantile utilities from
`utils.py` and adds the helpers shared by every distribution-aware model
(ARIMA, ETS, CES, Theta).
"""

import numpy as np
from scipy import stats as _scipy_stats  # aliased to avoid collision with the Distribution enum

from ._lib import distributions as _lib_dist
from .utils import (
    ArimaMethod,
    Distribution,
    _VALID_DISTRIBUTIONS,
    _calculate_intervals,
    _quantiles,
)

__all__ = [
    "Distribution",
    "ArimaMethod",
    "VALID_DISTRIBUTIONS",
    "distribution_n_extra_params",
    "switch_distribution",
    "dist_init_params",
    "dist_tail_bounds",
    "guard_dist_tail",
    "extract_dist_params",
    "aic_bic_aicc",
    "error_params_from_model",
    "_quantiles",
    "_calculate_intervals",
    "frozen_error_distribution",
]

VALID_DISTRIBUTIONS = _VALID_DISTRIBUTIONS


def distribution_n_extra_params(distribution: str) -> int:
    """Number of distribution params appended to the optimizer vector."""
    return 0 if distribution in ("laplace", "normal") else 2


def switch_distribution(distribution: str, module):
    """Map a distribution string to a C++ module's Distribution enum.

    `module` is one of statsforecast._lib.{ets, ces, theta}.
    """
    mapping = {
        "normal": "Normal",
        "laplace": "Laplace",
        "t": "StudentT",
        "skew-normal": "SkewNormal",
        "ged": "GED",
    }
    if distribution not in mapping:
        raise ValueError(f"Unknown distribution: {distribution!r}")
    return getattr(module.Distribution, mapping[distribution])


def dist_init_params(distribution: str, var_init: float):
    """Return (n_dist, dist_init) for the optimizer-vector tail.

    Layout (must match include/statsforecast/distributions.h):
      laplace     -> (0, [])
      t           -> (2, [log(var), log(3.0)])           # nu_init = 5
      skew-normal -> (2, [log(var), 0.0])
      ged         -> (2, [0.5*log(var), log(2.0)])       # GED stores log(sigma)
    """
    if distribution == "t":
        return 2, [np.log(var_init), np.log(3.0)]
    if distribution == "skew-normal":
        return 2, [np.log(var_init), 0.0]
    if distribution == "ged":
        return 2, [0.5 * np.log(var_init), np.log(2.0)]
    return 0, []  # laplace / normal


# Numerically safe box for the optimizer tail, as {name: ((lo, hi), (lo, hi))}
# for [log_scale, shape].  The limits themselves are defined once in
# include/statsforecast/distributions.h -- where the C++ likelihood cores clamp
# with them -- and read from here, so the two sides cannot drift apart.  Cached
# at import because guard_dist_tail() runs in the optimizer's inner loop.
_DIST_TAIL_BOUNDS = {
    str(name): tuple(zip(*_lib_dist.tail_bounds(switch_distribution(name, _lib_dist))))
    for name in _VALID_DISTRIBUTIONS
}
_TAIL_PENALTY_SCALE = 1.0


def dist_tail_bounds(distribution, n_dist: int = 2):
    """(lower, upper) arrays for the optimizer tail, for box-constrained solvers."""
    bounds = _DIST_TAIL_BOUNDS.get(str(distribution))
    if bounds is None or n_dist == 0:
        return np.full(n_dist, -np.inf), np.full(n_dist, np.inf)
    return (
        np.array([b[0] for b in bounds[:n_dist]]),
        np.array([b[1] for b in bounds[:n_dist]]),
    )


def guard_dist_tail(distribution, tail):
    """Project the optimizer tail into its numerically safe box.

    Returns `(safe_tail, penalty)`. `penalty` is 0.0 inside the box and grows
    quadratically outside it, so an unbounded optimizer that proposes a wild step
    gets a large *finite* objective whose gradient points back into the feasible
    region, instead of an OverflowError / ZeroDivisionError.

    Normal and Laplace have no tail and an unbounded box, so this is the
    identity for them.

    Precondition: `tail` is finite (callers reject non-finite trial points).
    """
    bounds = _DIST_TAIL_BOUNDS.get(str(distribution))
    if bounds is None:
        return list(tail), 0.0
    safe = []
    penalty = 0.0
    for value, (lo, hi) in zip(tail, bounds):
        value = float(value)
        if value < lo:
            penalty += (value - lo) ** 2
            value = lo
        elif value > hi:
            penalty += (value - hi) ** 2
            value = hi
        safe.append(value)
    return safe, _TAIL_PENALTY_SCALE * penalty


def extract_dist_params(distribution: str, fit_par_dist, residuals=None) -> dict:
    """Convert the fitted optimizer tail into model-dict keys.

    Returns a dict with `sigma2` plus the shape key for the distribution:
      t           -> {"nu", "sigma2"}
      skew-normal -> {"alpha_dist", "sigma2"}
      ged         -> {"beta_dist", "sigma2"}     # sigma2 = exp(log_sigma)**2
      laplace     -> {"sigma2"}                  # from residuals; b_hat = mean(|e|)
      normal      -> {}
    """
    if distribution in ("t", "skew-normal", "ged"):
        fit_par_dist, _ = guard_dist_tail(distribution, fit_par_dist)
    if distribution == "t":
        return {
            "nu": float(np.exp(fit_par_dist[1]) + 2.0),
            "sigma2": float(np.exp(fit_par_dist[0])),
        }
    if distribution == "skew-normal":
        return {
            "alpha_dist": float(fit_par_dist[1]),
            "sigma2": float(np.exp(fit_par_dist[0])),
        }
    if distribution == "ged":
        return {
            "beta_dist": float(np.exp(fit_par_dist[1])),
            "sigma2": float(np.exp(fit_par_dist[0])) ** 2,
        }
    if distribution == "laplace":
        b_hat = float(np.nanmean(np.abs(residuals)))
        return {"sigma2": 2.0 * b_hat ** 2}
    return {}


def error_params_from_model(model: dict):
    """Map a fitted model dict's distribution params to sample_errors() params.

    Returns a dict of extra kwargs for sample_errors(), or None if the
    distribution needs no extra params (normal / laplace).
    """
    dist = model.get("distribution", "normal")
    if dist == "t":
        return {"df": model["nu"]}
    if dist == "skew-normal":
        return {"skewness": model["alpha_dist"]}
    if dist == "ged":
        return {"shape": model["beta_dist"]}
    return None  # normal / laplace -> sample_errors derives scale from sigma


def aic_bic_aicc(neg2logL: float, np_eff: int, n: int):
    """Standard information criteria from -2*logLik and effective param count."""
    aic = neg2logL + 2 * np_eff
    bic = neg2logL + np.log(n) * np_eff
    if n - np_eff - 1 != 0.0:
        aicc = aic + 2 * np_eff * (np_eff + 1) / (n - np_eff - 1)
    else:
        aicc = np.inf
    return aic, bic, aicc


def frozen_error_distribution(sigma: float, distribution: str, params=None):
    """Return a scipy frozen distribution with sigma as the scale parameter.

    sigma  = sqrt(model["sigma2"]) — the MLE-fitted scale, NOT the SD.
    params = human-readable keys: {"df": nu}, {"skewness": alpha}, {"shape": beta}.

    Usage:
        d = frozen_error_distribution(sigma, "t", {"df": 5})
        d.ppf(0.975)          # analytic upper quantile
        d.rvs(100, rng)       # 100 MC samples
    """
    p = params or {}
    if distribution == "t":
        return _scipy_stats.t(df=p.get("df", 5.0), scale=sigma)
    if distribution == "laplace":
        return _scipy_stats.laplace(scale=sigma / np.sqrt(2))
    if distribution == "skew-normal":
        return _scipy_stats.skewnorm(a=p.get("skewness", 0.0), scale=sigma)
    if distribution == "ged":
        return _scipy_stats.gennorm(beta=p.get("shape", 2.0), scale=sigma)
    return _scipy_stats.norm(scale=sigma)  # normal (default)
