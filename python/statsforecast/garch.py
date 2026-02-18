__all__ = ['garch_model', 'garch_forecast']


import numpy as np
from scipy.optimize import minimize

from ._lib import garch as _garch


def generate_garch_data(n, w, alpha, beta):
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    p = len(alpha)
    q = len(beta)

    if w < 0 or any(alpha < 0) or any(beta < 0):
        raise ValueError("Coefficients must be nonnegative")
    if sum(alpha) + sum(beta) >= 1:
        raise ValueError(
            "Sum of coefficients of lagged versions of the series and lagged "
            "versions of volatility must be less than 1"
        )

    y = np.zeros(n)
    sigma2 = np.zeros(n)
    np.random.seed(1)

    if q != 0:
        sigma2[:q] = 1.0
    y[:p] = np.random.randn(p)

    alpha_rev = alpha[::-1]
    beta_rev = beta[::-1]

    start = max(p, q)
    for k in range(start, n):
        psum = np.sum(alpha_rev * y[k - p:k] ** 2)
        result = w + psum
        if q != 0:
            qsum = np.sum(beta_rev * sigma2[k - q:k])
            result += qsum
        sigma2[k] = result
        y[k] = np.random.randn() * np.sqrt(sigma2[k])
    return y


def garch_sigma2(x0, x, p, q):
    return _garch.compute_sigma2(np.asarray(x0, dtype=np.float64),
                                 np.asarray(x, dtype=np.float64), p, q)


def garch_loglik(x0, x, p, q):
    return _garch.loglik(np.asarray(x0, dtype=np.float64),
                         np.asarray(x, dtype=np.float64), p, q)


def garch_cons(x0):
    return _garch.constraint_value(np.asarray(x0, dtype=np.float64))


def garch_model(x, p, q):
    x = np.asarray(x, dtype=np.float64)
    n_params = p + q + 1
    x0 = np.full(n_params, 0.1)
    bounds = [(1e-8, None)] * n_params
    constraints = {"type": "ineq", "fun": garch_cons}

    def obj(params):
        return _garch.loglik(params, x, p, q)

    result = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    coeff = result.x
    sigma2 = garch_sigma2(coeff, x, p, q)

    np.random.seed(1)
    fitted = np.full(len(x), np.nan)
    for k in range(p, len(x)):
        error = np.random.randn()
        fitted[k] = error * np.sqrt(sigma2[k])

    res = {
        "p": p,
        "q": q,
        "coeff": coeff,
        "message": result.message,
        "y_vals": x[-p:],
        "sigma2_vals": sigma2[-q:] if q > 0 else np.array([]),
        "fitted": fitted,
    }

    return res


def garch_forecast(mod, h):
    coeff = mod["coeff"]
    p = mod["p"]
    q = mod["q"]
    y_vals = np.asarray(mod["y_vals"], dtype=np.float64)
    sigma2_vals = np.asarray(mod["sigma2_vals"], dtype=np.float64) if q > 0 else np.array([], dtype=np.float64)

    mean_vals, sigma2_out = _garch.forecast(coeff, p, q, h, y_vals, sigma2_vals)

    res = {"mean": mean_vals, "sigma2": sigma2_out, "fitted": mod["fitted"]}

    return res
