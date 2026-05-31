__all__ = ['ucm_model', 'ucm_forecast']

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple

# Diffuse initialization: initial state mean is 0 and the initial state
# covariance is a large multiple of the identity.
_DIFFUSE_VARIANCE = 1e6

def _build_matrices(
    season_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the time-invariant state space matrices for the UCM."""
    s = int(season_length)
    has_seasonal = s > 1
    n_seas = s - 1 if has_seasonal else 0
    k = 2 + n_seas

    # Observation matrix: y_t = mu_t + gamma_t + eps_t.
    Z = np.zeros((1, k))
    Z[0, 0] = 1.0  # level
    if has_seasonal:
        Z[0, 2] = 1.0  # current seasonal effect

    # Transition matrix.
    T = np.zeros((k, k))
    T[0, 0] = 1.0  # mu_{t+1} = mu_t + beta_t
    T[0, 1] = 1.0
    T[1, 1] = 1.0  # beta_{t+1} = beta_t
    if has_seasonal:
        # gamma_t = -(gamma_{t-1} + ... + gamma_{t-s+1}) + omega_t
        T[2, 2 : 2 + n_seas] = -1.0
        # The remaining seasonal states are lagged copies (companion form).
        for i in range(1, n_seas):
            T[2 + i, 2 + i - 1] = 1.0

    # State-noise selection matrix: shocks enter the level, slope and (if
    # present) the current seasonal state.
    n_shocks = 3 if has_seasonal else 2
    R = np.zeros((k, n_shocks))
    R[0, 0] = 1.0  # level shock
    R[1, 1] = 1.0  # slope shock
    if has_seasonal:
        R[2, 2] = 1.0  # seasonal shock

    return Z, T, R


def kalman_filter(
    y: np.ndarray,
    Z: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    H: float,
    a0: np.ndarray,
    P0: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run a univariate Kalman filter."""
    n = y.shape[0]
    z = Z[0]
    RQRt = R @ Q @ R.T

    a_pred = a0.astype(np.float64).copy()
    P_pred = P0.astype(np.float64).copy()

    one_step_pred = np.empty(n, dtype=np.float64)
    loglik = 0.0
    log_2pi = np.log(2.0 * np.pi)

    for t in range(n):
        # Forecast for the current observation.
        y_hat = z @ a_pred
        one_step_pred[t] = y_hat
        v = y[t] - y_hat
        Pz = P_pred @ z
        F = z @ Pz + H

        # Update step (skip if F is not usable, e.g. during diffuse start).
        if np.isfinite(F) and F > 0:
            loglik += -0.5 * (log_2pi + np.log(F) + v * v / F)
            K = Pz / F
            a_filt = a_pred + K * v
            P_filt = P_pred - np.outer(K, Pz)
        else:
            a_filt = a_pred
            P_filt = P_pred

        # Predict the next state.
        a_pred = T @ a_filt
        P_pred = T @ P_filt @ T.T + RQRt

    return loglik, a_filt, one_step_pred


def _make_QH(theta: np.ndarray, has_seasonal: bool) -> Tuple[np.ndarray, float]:
    """Build the state-noise covariance Q and observation variance H."""
    H = theta[0]
    if has_seasonal:
        Q = np.diag(theta[1:4])
    else:
        Q = np.diag(theta[1:3])
    return Q, H


def _nll(
    theta: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
    has_seasonal: bool,
) -> float:
    """Negative log-likelihood for a variance vector theta."""
    Q, H = _make_QH(theta, has_seasonal)
    loglik, _, _ = kalman_filter(y, Z, T, R, Q, H, a0, P0)
    if not np.isfinite(loglik):
        return 1e10
    return -loglik


def ucm_model(y: np.ndarray, season_length: int = 1) -> Dict:
    """Fit the minimal UCM by maximum likelihood."""
    y = np.asarray(y, dtype=np.float64)
    s = int(season_length)
    has_seasonal = s > 1

    Z, T, R = _build_matrices(s)
    k = T.shape[0]
    a0 = np.zeros(k)
    P0 = _DIFFUSE_VARIANCE * np.eye(k)

    # Starting values and bounds: variances scaled by the series variance for
    # numerical conditioning across different data scales.
    var_y = float(np.var(y))
    if var_y <= 0 or not np.isfinite(var_y):
        var_y = 1.0
    n_params = 4 if has_seasonal else 3
    x0 = np.full(n_params, var_y / 10.0)
    bounds = [(1e-8 * var_y, None)] * n_params

    result = minimize(
        _nll,
        x0,
        args=(y, Z, T, R, a0, P0, has_seasonal),
        method="L-BFGS-B",
        bounds=bounds,
    )
    params = result.x

    Q, H = _make_QH(params, has_seasonal)
    _, a_n, fitted = kalman_filter(y, Z, T, R, Q, H, a0, P0)

    residuals = y - fitted
    sigma = np.sqrt(np.sum(residuals**2) / max(len(y) - 1, 1))

    return {
        "params": params,
        "Z": Z,
        "T": T,
        "R": R,
        "Q": Q,
        "H": H,
        "a_n": a_n,
        "fitted": fitted,
        "sigma": sigma,
        "season_length": s,
        "message": result.message,
    }


def ucm_forecast(mod: Dict, h: int) -> Dict:
    """Produce h-step-ahead point forecasts from a fitted UCM."""
    Z = mod["Z"]
    T = mod["T"]
    z = Z[0]
    a = mod["a_n"].astype(np.float64).copy()

    mean = np.empty(h, dtype=np.float64)
    for i in range(h):
        a = T @ a
        mean[i] = z @ a

    return {"mean": mean, "fitted": mod["fitted"]}
