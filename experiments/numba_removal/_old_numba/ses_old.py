"""Old numba SES and intermittent demand functions from models.py"""

import numpy as np
from numba import njit


@njit(cache=True)
def _ses_sse(alpha, x):
    complement = 1 - alpha
    forecast = x[0]
    sse = 0.0
    for i in range(1, len(x)):
        forecast = alpha * x[i - 1] + complement * forecast
        sse += (x[i] - forecast) ** 2
    return sse


@njit(cache=True)
def _ses_forecast(x, alpha):
    complement = 1 - alpha
    fitted = np.empty_like(x)
    fitted[0] = x[0]
    j = 0
    for i in range(1, len(x)):
        fitted[i] = alpha * x[j] + complement * fitted[j]
        j += 1
    forecast = alpha * x[j] + complement * fitted[j]
    fitted[0] = np.nan
    return forecast, fitted


@njit(cache=True)
def _expand_fitted_demand(fitted, y):
    out = np.empty_like(y)
    out[0] = np.nan
    fitted_idx = 0
    for i in range(1, y.size):
        if y[i - 1] > 0:
            fitted_idx += 1
            out[i] = fitted[fitted_idx]
        elif fitted_idx > 0:
            out[i] = out[i - 1]
        else:
            out[i] = y[i - 1]
    return out


@njit(cache=True)
def _expand_fitted_intervals(fitted, y):
    out = np.empty_like(y)
    out[0] = np.nan
    fitted_idx = 0
    for i in range(1, y.size):
        if y[i - 1] != 0:
            fitted_idx += 1
            if fitted[fitted_idx] == 0:
                out[i] = 1
            else:
                out[i] = fitted[fitted_idx]
        elif fitted_idx > 0:
            out[i] = out[i - 1]
        else:
            out[i] = 1
    return out
