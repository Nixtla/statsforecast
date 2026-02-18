"""Benchmark: ARIMA CSS optimization with analytical vs numerical gradients."""

import math
import time

import numpy as np
from scipy.optimize import minimize

from statsforecast.arima import (
    arima,
    arima_css,
    arima_css_grad,
    arima_transpar,
    transpar_vjp,
)
from statsforecast.utils import AirPassengers as ap


def bench_case(label, y, order, seasonal=None, include_mean=False, n_reps=50):
    if seasonal is None:
        seasonal = {"order": (0, 0, 0), "period": 1}

    # --- Setup: run arima once to get initial params and problem structure ---
    result = arima(y, order=order, seasonal=seasonal, include_mean=include_mean, method="CSS")
    p_order, d_order, q_order = order
    sp, sd, sq = seasonal["order"]
    ns = seasonal["period"]

    arma_arr = np.array(
        [p_order, q_order, sp, sq, ns, d_order, sd], dtype=np.int32
    )
    narma = p_order + q_order + sp + sq

    # Reconstruct xreg
    nd = d_order + sd
    ncxreg = 0
    xreg = None
    if include_mean and nd == 0:
        xreg = np.ones((len(y), 1), dtype=np.float64)
        ncxreg = 1
    xreg_idx = narma + np.arange(ncxreg) if ncxreg > 0 else None

    ncond = d_order + sd * ns + p_order + sp * ns

    # Extract fitted coefficients as starting point
    coef_names = list(result["coef"].keys())
    coef_vals = np.array([float(v) for v in result["coef"].values()])
    n_params = narma + ncxreg
    mask = np.ones(n_params, dtype=bool)

    # Perturb starting point away from optimum to force real optimization work
    rng = np.random.RandomState(42)
    init = coef_vals + rng.randn(n_params) * 0.1

    # --- Define value-only objective (old: numerical gradients) ---
    def css_val_only(p, x):
        par = np.zeros(n_params)
        par[mask] = p
        phi, theta = arima_transpar(par, arma_arr, False)
        x_adj = x.copy()
        if ncxreg > 0:
            x_adj = x_adj - np.dot(xreg, par[xreg_idx])
        res, _ = arima_css(x_adj, arma_arr, phi, theta)
        if math.isinf(res) or res <= 0:
            return 1e308 if math.isinf(res) else -math.inf
        return 0.5 * math.log(res)

    # --- Define value+gradient objective (new: analytical gradients) ---
    def css_val_and_grad(p, x):
        par = np.zeros(n_params)
        par[mask] = p
        phi, theta = arima_transpar(par, arma_arr, False)
        x_adj = x.copy()
        if ncxreg > 0:
            x_adj = x_adj - np.dot(xreg, par[xreg_idx])

        sigma2, resid, d_ssq_d_phi, d_ssq_d_theta, d_ssq_d_y = arima_css_grad(
            x_adj, arma_arr, phi, theta
        )
        if math.isinf(sigma2) or sigma2 <= 0:
            val = 1e308 if math.isinf(sigma2) else -math.inf
            return val, np.zeros(mask.sum())

        nu = int(np.count_nonzero(~np.isnan(resid[ncond:])))
        ssq = sigma2 * nu
        val = 0.5 * math.log(sigma2)

        bar_arma = transpar_vjp(d_ssq_d_phi, d_ssq_d_theta, par[:narma], arma_arr)
        full_grad = np.zeros(n_params)
        full_grad[:narma] = bar_arma
        if ncxreg > 0:
            full_grad[xreg_idx] = -d_ssq_d_y @ xreg
        grad = full_grad[mask] / (2.0 * ssq)
        return val, grad

    optim_opts = {"maxiter": 100}

    # --- Verify both converge to same result ---
    res_num = minimize(css_val_only, init, args=(y,), method="BFGS",
                       tol=1e-8, options=optim_opts)
    res_ana = minimize(css_val_and_grad, init, args=(y,), method="BFGS",
                       jac=True, tol=1e-8, options=optim_opts)

    param_match = np.allclose(res_num.x, res_ana.x, atol=1e-4)
    val_match = np.allclose(res_num.fun, res_ana.fun, atol=1e-8)

    # --- Benchmark ---
    # Warmup
    for _ in range(3):
        minimize(css_val_only, init, args=(y,), method="BFGS",
                 tol=1e-8, options=optim_opts)
        minimize(css_val_and_grad, init, args=(y,), method="BFGS",
                 jac=True, tol=1e-8, options=optim_opts)

    t0 = time.perf_counter()
    for _ in range(n_reps):
        minimize(css_val_only, init, args=(y,), method="BFGS",
                 tol=1e-8, options=optim_opts)
    t_num = (time.perf_counter() - t0) / n_reps

    t0 = time.perf_counter()
    for _ in range(n_reps):
        minimize(css_val_and_grad, init, args=(y,), method="BFGS",
                 jac=True, tol=1e-8, options=optim_opts)
    t_ana = (time.perf_counter() - t0) / n_reps

    n_p = len(init)
    print(f"  {label} (N={len(y)}, params={n_p})")
    print(f"    numerical:  {t_num*1000:8.2f} ms  (nit={res_num.nit}, nfev={res_num.nfev})")
    print(f"    analytical: {t_ana*1000:8.2f} ms  (nit={res_ana.nit}, nfev={res_ana.nfev})")
    print(f"    speedup:    {t_num/t_ana:.2f}x")
    print(f"    params match: {param_match}  val match: {val_match}")
    print()


def main():
    print("=" * 65)
    print("  ARIMA CSS Optimization: Analytical vs Numerical Gradients")
    print("=" * 65)
    print()

    # Case 1: Simple ARIMA(1,1,1) on AirPassengers (2 params)
    bench_case("ARIMA(1,1,1)", ap.astype(np.float64), order=(1, 1, 1))

    # Case 2: ARIMA(2,1,2) on AirPassengers (4 params)
    bench_case("ARIMA(2,1,2)", ap.astype(np.float64), order=(2, 1, 2))

    # Case 3: ARIMA(3,1,3) (6 params)
    bench_case("ARIMA(3,1,3)", ap.astype(np.float64), order=(3, 1, 3))

    # Case 4: ARIMA(1,0,1) with mean (3 params)
    np.random.seed(42)
    y_stationary = np.random.randn(500).astype(np.float64) + 5.0
    bench_case("ARIMA(1,0,1)+mean, n=500", y_stationary,
               order=(1, 0, 1), include_mean=True)

    # Case 5: Seasonal ARIMA(1,0,1)(1,0,1)[12] (4 params)
    np.random.seed(42)
    y_seasonal = np.random.randn(300).astype(np.float64)
    bench_case("SARIMA(1,0,1)(1,0,1)[12]", y_seasonal,
               order=(1, 0, 1),
               seasonal={"order": (1, 0, 1), "period": 12})

    # Case 6: Seasonal ARIMA(2,1,1)(1,1,1)[12] on AirPassengers (5 params)
    bench_case("SARIMA(2,1,1)(1,1,1)[12]", ap.astype(np.float64),
               order=(2, 1, 1),
               seasonal={"order": (1, 1, 1), "period": 12})

    # Case 7: Longer series ARIMA(2,1,2) n=2000 (4 params)
    np.random.seed(42)
    y_long = np.cumsum(np.random.randn(2000)).astype(np.float64)
    bench_case("ARIMA(2,1,2), n=2000", y_long, order=(2, 1, 2))

    # Case 8: Longer series ARIMA(3,1,3) n=5000 (6 params)
    np.random.seed(42)
    y_vlong = np.cumsum(np.random.randn(5000)).astype(np.float64)
    bench_case("ARIMA(3,1,3), n=5000", y_vlong, order=(3, 1, 3), n_reps=20)


if __name__ == "__main__":
    main()
