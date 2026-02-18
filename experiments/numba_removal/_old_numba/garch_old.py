"""Old numba GARCH functions from garch.py"""

import numpy as np
from numba import njit


@njit(cache=True)
def generate_garch_data(n, w, alpha, beta):
    np.random.seed(1)
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    p = len(alpha)
    q = len(beta)

    w_vals = w < 0
    alpha_vals = np.any(alpha < 0)
    beta_vals = np.any(beta < 0)

    if np.any(np.array([w_vals, alpha_vals, beta_vals])):
        raise ValueError("Coefficients must be nonnegative")

    if np.sum(alpha) + np.sum(beta) >= 1:
        raise ValueError(
            "Sum of coefficients of lagged versions of the series and lagged "
            "versions of volatility must be less than 1"
        )

    if q != 0:
        sigma2[0:q] = 1

    for k in range(p):
        y[k] = np.random.normal(loc=0, scale=1)

    for k in range(max(p, q), n):
        psum = np.flip(alpha) * (y[k - p : k] ** 2)
        psum = np.nansum(psum)
        if q != 0:
            qsum = np.flip(beta) * (sigma2[k - q : k])
            qsum = np.nansum(qsum)
            sigma2[k] = w + psum + qsum
        else:
            sigma2[k] = w + psum
        y[k] = np.random.normal(loc=0, scale=np.sqrt(sigma2[k]))

    return y


@njit(cache=True)
def garch_sigma2(x0, x, p, q):
    w = x0[0]
    alpha = x0[1 : (p + 1)]
    beta = x0[(p + 1) :]

    sigma2 = np.full((len(x),), np.nan)
    sigma2[0] = np.var(x)

    for k in range(max(p, q), len(x)):
        psum = np.flip(alpha) * (x[k - p : k] ** 2)
        psum = np.nansum(psum)
        if q != 0:
            qsum = np.flip(beta) * (sigma2[k - q : k])
            qsum = np.nansum(qsum)
            sigma2[k] = w + psum + qsum
        else:
            sigma2[k] = w + psum

    return sigma2


@njit(cache=True)
def garch_cons(x0):
    return 1 - (x0[1:].sum())


@njit(cache=True)
def garch_loglik(x0, x, p, q):
    sigma2 = garch_sigma2(x0, x, p, q)
    z = x - np.nanmean(x)
    loglik = 0

    for k in range(max(p, q), len(z)):
        if sigma2[k] == 0:
            sigma2[k] = 1e-10
        loglik = loglik - 0.5 * (
            np.log(2 * np.pi) + np.log(sigma2[k]) + (z[k] ** 2) / sigma2[k]
        )

    return -loglik
