"""Old numba MFLES functions from mfles.py"""

import numpy as np
from numba import njit


@njit(cache=True)
def get_basis(y, n_changepoints, decay=-1, gradient_strategy=0):
    if n_changepoints < 1:
        return np.arange(y.size, dtype=np.float64).reshape(-1, 1)
    y = y.copy()
    y -= y[0]
    n = len(y)
    if gradient_strategy:
        gradients = np.abs(y[:-1] - y[1:])
    initial_point = y[0]
    final_point = y[-1]
    mean_y = np.mean(y)
    changepoints = np.empty(shape=(len(y), n_changepoints + 1))
    array_splits = []
    for i in range(1, n_changepoints + 1):
        i = n_changepoints - i + 1
        if gradient_strategy:
            cps = np.argsort(-gradients)
            cps = cps[cps > 0.1 * len(gradients)]
            cps = cps[cps < 0.9 * len(gradients)]
            split_point = cps[i - 1]
            array_splits.append(y[:split_point])
        else:
            split_point = len(y) // i
            array_splits.append(y[:split_point])
            y = y[split_point:]
    len_splits = 0
    for i in range(n_changepoints):
        if gradient_strategy:
            len_splits = len(array_splits[i])
        else:
            len_splits += len(array_splits[i])
        moving_point = array_splits[i][-1]
        left_basis = np.linspace(initial_point, moving_point, len_splits)
        if decay is None:
            end_point = final_point
        else:
            if decay == -1:
                dd = moving_point**2
                if mean_y != 0:
                    dd /= mean_y**2
                if dd > 0.99:
                    dd = 0.99
                if dd < 0.001:
                    dd = 0.001
                end_point = moving_point - ((moving_point - final_point) * (1 - dd))
            else:
                end_point = moving_point - ((moving_point - final_point) * (1 - decay))
        right_basis = np.linspace(moving_point, end_point, n - len_splits + 1)
        changepoints[:, i] = np.append(left_basis, right_basis[1:])
    changepoints[:, i + 1] = np.ones(n)
    return changepoints


@njit(cache=True)
def siegel_repeated_medians(x, y):
    n = y.size
    slopes = np.empty_like(y)
    slopes_sub = np.empty(shape=n - 1, dtype=y.dtype)
    for i in range(n):
        k = 0
        for j in range(n):
            if i == j:
                continue
            xd = x[j] - x[i]
            if xd == 0:
                slope = 0
            else:
                slope = (y[j] - y[i]) / xd
            slopes_sub[k] = slope
            k += 1
        slopes[i] = np.median(slopes_sub)
    ints = y - slopes * x
    return x * np.median(slopes) + np.median(ints)
