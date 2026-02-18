"""Old numba TBATS functions from tbats.py"""

import numpy as np
from numba import njit


@njit(cache=True)
def makeTBATSFMatrix(
    phi, tau, alpha, beta, ar_coeffs, ma_coeffs, gamma_bold, seasonal_periods, k_vector
):
    # Alpha row
    F = np.array([[1.0]])
    if phi is not None:
        F = np.hstack((F, np.array([[phi]])))
    F = np.hstack((F, np.zeros((1, tau))))
    if ar_coeffs is not None:
        alpha_varphi = alpha * ar_coeffs
        alpha_varphi = alpha_varphi.reshape((1, len(alpha_varphi)))
        F = np.hstack((F, alpha_varphi))
    if ma_coeffs is not None:
        alpha_theta = alpha * ma_coeffs
        alpha_theta = alpha_theta.reshape((1, len(alpha_theta)))
        F = np.hstack((F, alpha_theta))

    # Beta row
    if beta is not None:
        beta_row = np.array([[0.0, phi]])
        beta_row = np.hstack((beta_row, np.zeros((1, tau))))
        if ar_coeffs is not None:
            beta_varphi = beta * ar_coeffs
            beta_varphi = beta_varphi.reshape((1, len(beta_varphi)))
            beta_row = np.hstack((beta_row, beta_varphi))
        if ma_coeffs is not None:
            beta_theta = beta * ma_coeffs
            beta_theta = beta_theta.reshape((1, len(beta_theta)))
            beta_row = np.hstack((beta_row, beta_theta))
        F = np.vstack((F, beta_row))

    # Seasonal row
    seasonal_row = np.zeros((tau, 1))
    if phi is not None:
        seasonal_row = np.hstack((seasonal_row, np.zeros((tau, 1))))

    A = np.zeros((tau, tau))
    pos = 0
    for k, period in zip(k_vector, seasonal_periods):
        t = 2 * np.pi * np.arange(1, k + 1) / period
        ck = np.diag(np.cos(t))
        sk = np.diag(np.sin(t))
        top = np.hstack((ck, sk))
        bottom = np.hstack((-sk, ck))
        Ak = np.vstack((top, bottom))
        A[pos : pos + 2 * k, pos : pos + 2 * k] = Ak
        pos += 2 * k
    seasonal_row = np.hstack((seasonal_row, A))

    if ar_coeffs is not None:
        varphi = ar_coeffs.reshape((1, ar_coeffs.shape[0]))
        B = np.dot(np.transpose(gamma_bold), varphi)
        seasonal_row = np.hstack((seasonal_row, B))

    if ma_coeffs is not None:
        theta = ma_coeffs.reshape((1, ma_coeffs.shape[0]))
        C = np.dot(np.transpose(gamma_bold), theta)
        seasonal_row = np.hstack((seasonal_row, C))

    F = np.vstack((F, seasonal_row))

    # ARMA submatrix
    if ar_coeffs is not None:
        p = len(ar_coeffs)
        ar_rows = np.zeros((p, 1))
        if phi is not None:
            ar_rows = np.hstack((ar_rows, ar_rows))
        ar_seasonal_zeros = np.zeros((p, tau))
        ar_rows = np.hstack((ar_rows, ar_seasonal_zeros))
        ident = np.eye(p - 1)
        ident = np.hstack((ident, np.zeros(((p - 1), 1))))
        ar_part = np.vstack((ar_coeffs.reshape((1, ar_coeffs.shape[0])), ident))
        ar_rows = np.hstack((ar_rows, ar_part))
        if ma_coeffs is not None:
            q = len(ma_coeffs)
            ma_in_ar = np.zeros((p, q))
            ma_in_ar[0, :] = ma_coeffs
            ar_rows = np.hstack((ar_rows, ma_in_ar))
        F = np.vstack((F, ar_rows))

    if ma_coeffs is not None:
        q = len(ma_coeffs)
        ma_rows = np.zeros((q, 1))
        if phi is not None:
            ma_rows = np.hstack((ma_rows, ma_rows))
        ma_seasonal = np.zeros((q, tau))
        ma_rows = np.hstack((ma_rows, ma_seasonal))
        if ar_coeffs is not None:
            p = len(ar_coeffs)
            ar_in_ma = np.zeros((q, p))
            ma_rows = np.hstack((ma_rows, ar_in_ma))
        ident = np.eye(q - 1)
        ident = np.hstack((ident, np.zeros(((q - 1), 1))))
        ma_part = np.vstack((np.zeros((1, q)), ident))
        ma_rows = np.hstack((ma_rows, ma_part))
        F = np.vstack((F, ma_rows))

    return F


@njit(cache=True)
def calcTBATSFaster(y_trans, w_transpose, g, F, x_nought):
    n = y_trans.shape[0]

    yhat = np.zeros((1, n))
    e = np.zeros((1, n))
    x = np.zeros((n, len(x_nought)))

    yhat[0, 0] = np.dot(w_transpose[0], x_nought)
    e[0, 0] = y_trans[0] - yhat[0, 0]
    x[0] = np.dot(F, x_nought) + (g[:, 0] * e[0, 0])

    for j in range(1, y_trans.shape[0]):
        yhat[:, j] = np.dot(w_transpose[0], x[j - 1])
        e[0, j] = y_trans[j] - yhat[0, j]
        x[j] = np.dot(F, x[j - 1]) + g[:, 0] * e[0, j]

    return yhat, e, x
