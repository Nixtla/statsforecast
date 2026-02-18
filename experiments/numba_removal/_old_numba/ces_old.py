"""Old numba CES functions from ces.py"""

import math

import numpy as np
from numba import njit

NONE = 0
SIMPLE = 1
PARTIAL = 2
FULL = 3
TOL = 1.0e-10
NA = -99999.0


@njit(cache=True)
def cesupdate(states, i, m, season, alpha_0, alpha_1, beta_0, beta_1, y):
    if season in (NONE, PARTIAL, FULL):
        e = y - states[i - 1, 0]
    else:
        e = y - states[i - m, 0]
    if season > SIMPLE:
        e -= states[i - m, 2]

    if season in (NONE, PARTIAL, FULL):
        states[i, 0] = (
            states[i - 1, 0]
            - (1.0 - alpha_1) * states[i - 1, 1]
            + (alpha_0 - alpha_1) * e
        )
        states[i, 1] = (
            states[i - 1, 0]
            + (1.0 - alpha_0) * states[i - 1, 1]
            + (alpha_0 + alpha_1) * e
        )
    else:
        states[i, 0] = (
            states[i - m, 0]
            - (1.0 - alpha_1) * states[i - m, 1]
            + (alpha_0 - alpha_1) * e
        )
        states[i, 1] = (
            states[i - m, 0]
            + (1.0 - alpha_0) * states[i - m, 1]
            + (alpha_0 + alpha_1) * e
        )

    if season == PARTIAL:
        states[i, 2] = states[i - m, 2] + beta_0 * e
    if season == FULL:
        states[i, 2] = (
            states[i - m, 2] - (1 - beta_1) * states[i - m, 3] + (beta_0 - beta_1) * e
        )
        states[i, 3] = (
            states[i - m, 2] + (1 - beta_0) * states[i - m, 3] + (beta_0 + beta_1) * e
        )


@njit(cache=True)
def cesfcst(states, i, m, season, f, h, alpha_0, alpha_1, beta_0, beta_1):
    new_states = np.zeros((m + h, states.shape[1]), dtype=np.float32)
    new_states[:m] = states[(i - m) : i]
    for i_h in range(m, m + h):
        if season in (NONE, PARTIAL, FULL):
            f[i_h - m] = new_states[i_h - 1, 0]
        else:
            f[i_h - m] = new_states[i_h - m, 0]
        if season > SIMPLE:
            f[i_h - m] += new_states[i_h - m, 2]
        cesupdate(
            new_states, i_h, m, season, alpha_0, alpha_1, beta_0, beta_1, f[i_h - m]
        )
    return new_states


@njit(cache=True)
def cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1,
            e, amse, nmse, backfit):
    denom = np.zeros(nmse)
    m = 1 if season == NONE else m
    f = np.zeros(max(nmse, m))
    lik = 0.0
    lik2 = 0.0
    amse[:nmse] = 0.0
    n = len(y)
    for i in range(m, n + m):
        cesfcst(states, i, m, season, f, nmse, alpha_0, alpha_1, beta_0, beta_1)
        if math.fabs(f[0] - NA) < TOL:
            lik = NA
            return lik
        e[i - m] = y[i - m] - f[0]
        for j in range(nmse):
            if (i + j) < n:
                denom[j] += 1.0
                tmp = y[i + j] - f[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + (tmp * tmp)) / denom[j]
        cesupdate(states, i, m, season, alpha_0, alpha_1, beta_0, beta_1, y[i - m])
        lik = lik + e[i - m] * e[i - m]
        lik2 += math.log(math.fabs(f[0]))
    new_states = cesfcst(
        states, n + m, m, season, f, m, alpha_0, alpha_1, beta_0, beta_1
    )
    states[-m:] = new_states[-m:]
    lik = n * math.log(lik)
    if not backfit:
        return lik
    y[:] = y[::-1]
    states[:] = states[::-1]
    e[:] = e[::-1]
    lik = 0.0
    lik2 = 0.0
    for i in range(m, n + m):
        cesfcst(states, i, m, season, f, nmse, alpha_0, alpha_1, beta_0, beta_1)
        if math.fabs(f[0] - NA) < TOL:
            lik = NA
            return lik
        e[i - m] = y[i - m] - f[0]
        for j in range(nmse):
            if (i + j) < n:
                denom[j] += 1.0
                tmp = y[i + j] - f[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + (tmp * tmp)) / denom[j]
        cesupdate(states, i, m, season, alpha_0, alpha_1, beta_0, beta_1, y[i - m])
        lik = lik + e[i - m] * e[i - m]
        lik2 += math.log(math.fabs(f[0]))
    new_states = cesfcst(
        states, n + m, m, season, f, m, alpha_0, alpha_1, beta_0, beta_1
    )
    states[-m:] = new_states[-m:]
    # fit again
    lik = 0.0
    lik2 = 0.0
    y[:] = y[::-1]
    states[:] = states[::-1]
    e[:] = e[::-1]
    for i in range(m, n + m):
        cesfcst(states, i, m, season, f, nmse, alpha_0, alpha_1, beta_0, beta_1)
        if math.fabs(f[0] - NA) < TOL:
            lik = NA
            return lik
        e[i - m] = y[i - m] - f[0]
        for j in range(nmse):
            if (i + j) < n:
                denom[j] += 1.0
                tmp = y[i + j] - f[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + (tmp * tmp)) / denom[j]
        cesupdate(states, i, m, season, alpha_0, alpha_1, beta_0, beta_1, y[i - m])
        lik = lik + e[i - m] * e[i - m]
        lik2 += math.log(math.fabs(f[0]))
    new_states = cesfcst(
        states, n + m, m, season, f, m, alpha_0, alpha_1, beta_0, beta_1
    )
    states[-m:] = new_states[-m:]
    lik = n * math.log(lik)
    return lik


@njit(cache=True)
def cesforecast(states, n, m, season, f, h, alpha_0, alpha_1, beta_0, beta_1):
    m = 1 if season == NONE else m
    new_states = cesfcst(
        states=states, i=m + n, m=m, season=season, f=f, h=h,
        alpha_0=alpha_0, alpha_1=alpha_1, beta_0=beta_0, beta_1=beta_1,
    )
    return new_states


@njit(cache=True)
def switch_ces(x):
    if x == "N":
        return 0
    elif x == "S":
        return 1
    elif x == "P":
        return 2
    elif x == "F":
        return 3
    return -1
