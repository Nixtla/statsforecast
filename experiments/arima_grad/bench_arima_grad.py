"""Benchmark: ARIMA CSS and ML optimization with analytical vs numerical gradients."""

import math
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

from statsforecast.arima import (
    arima,
    arima_css,
    arima_css_grad,
    arima_gradtrans,
    arima_like,
    arima_like_grad,
    arima_transpar,
    arima_undopars,
    getQ0,
    getQ0_vjp,
    make_arima,
    transpar_vjp,
)
from statsforecast.utils import AirPassengers as ap


def _minimize_force_numerical(fn, x0, args=(), method="BFGS", jac=None, **kw):
    """Wrapper around scipy.optimize.minimize that strips jac=True.

    When jac=True, the objective returns (val, grad). This wrapper extracts
    only the value, forcing BFGS to compute its own finite-difference gradients.
    """
    if jac is True:

        def val_only(*a, **ka):
            r = fn(*a, **ka)
            return r[0] if isinstance(r, tuple) else r

        return _scipy_minimize(val_only, x0, args=args, method=method, **kw)
    return _scipy_minimize(fn, x0, args=args, method=method, jac=jac, **kw)


# ---------------------------------------------------------------------------
# CSS benchmark (isolated minimize call)
# ---------------------------------------------------------------------------
def bench_css_case(label, y, order, seasonal=None, include_mean=False, n_reps=50):
    if seasonal is None:
        seasonal = {"order": (0, 0, 0), "period": 1}

    result = arima(
        y, order=order, seasonal=seasonal, include_mean=include_mean, method="CSS"
    )
    p_order, d_order, q_order = order
    sp, sd, sq = seasonal["order"]
    ns = seasonal["period"]

    arma_arr = np.array(
        [p_order, q_order, sp, sq, ns, d_order, sd], dtype=np.int32
    )
    narma = p_order + q_order + sp + sq

    nd = d_order + sd
    ncxreg = 0
    xreg = None
    if include_mean and nd == 0:
        xreg = np.ones((len(y), 1), dtype=np.float64)
        ncxreg = 1
    xreg_idx = narma + np.arange(ncxreg) if ncxreg > 0 else None

    ncond = d_order + sd * ns + p_order + sp * ns

    coef_vals = np.array([float(v) for v in result["coef"].values()])
    n_params = narma + ncxreg
    mask = np.ones(n_params, dtype=bool)

    rng = np.random.RandomState(42)
    init = coef_vals + rng.randn(n_params) * 0.1

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

    res_num = _scipy_minimize(
        css_val_only, init, args=(y,), method="BFGS", tol=1e-8, options=optim_opts
    )
    res_ana = _scipy_minimize(
        css_val_and_grad,
        init,
        args=(y,),
        method="BFGS",
        jac=True,
        tol=1e-8,
        options=optim_opts,
    )

    param_match = np.allclose(res_num.x, res_ana.x, atol=1e-4)
    val_match = np.allclose(res_num.fun, res_ana.fun, atol=1e-8)

    for _ in range(3):
        _scipy_minimize(
            css_val_only, init, args=(y,), method="BFGS", tol=1e-8, options=optim_opts
        )
        _scipy_minimize(
            css_val_and_grad,
            init,
            args=(y,),
            method="BFGS",
            jac=True,
            tol=1e-8,
            options=optim_opts,
        )

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _scipy_minimize(
            css_val_only, init, args=(y,), method="BFGS", tol=1e-8, options=optim_opts
        )
    t_num = (time.perf_counter() - t0) / n_reps

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _scipy_minimize(
            css_val_and_grad,
            init,
            args=(y,),
            method="BFGS",
            jac=True,
            tol=1e-8,
            options=optim_opts,
        )
    t_ana = (time.perf_counter() - t0) / n_reps

    n_p = len(init)
    print(f"  {label} (N={len(y)}, params={n_p})")
    print(
        f"    numerical:  {t_num*1000:8.2f} ms  (nit={res_num.nit}, nfev={res_num.nfev})"
    )
    print(
        f"    analytical: {t_ana*1000:8.2f} ms  (nit={res_ana.nit}, nfev={res_ana.nfev})"
    )
    print(f"    speedup:    {t_num/t_ana:.2f}x")
    print(f"    params match: {param_match}  val match: {val_match}")
    print()


# ---------------------------------------------------------------------------
# ML benchmark (isolated minimize call through Kalman filter)
# ---------------------------------------------------------------------------
def bench_ml_case(label, y, order, seasonal=None, include_mean=False, n_reps=50):
    if seasonal is None:
        seasonal = {"order": (0, 0, 0), "period": 1}

    css_result = arima(
        y, order=order, seasonal=seasonal, include_mean=include_mean, method="CSS"
    )
    p_order, d_order, q_order = order
    sp, sd, sq = seasonal["order"]
    ns = seasonal["period"]

    arma = np.array(
        [p_order, q_order, sp, sq, ns, d_order, sd], dtype=np.int32
    )
    narma = p_order + q_order + sp + sq

    nd = d_order + sd
    ncxreg = 0
    xreg = None
    if include_mean and nd == 0:
        xreg = np.ones((len(y), 1), dtype=np.float64)
        ncxreg = 1
    xreg_idx = narma + np.arange(ncxreg) if ncxreg > 0 else None

    coef_vals = np.array([float(v) for v in css_result["coef"].values()])
    n_params = narma + ncxreg
    mask_arr = np.ones(n_params, dtype=bool)
    coef = np.full(n_params, np.nan)

    Delta = np.array([1.0])
    for _ in range(d_order):
        Delta = np.convolve(Delta, [1.0, -1.0])
    for _ in range(sd):
        Delta = np.convolve(Delta, [1] + [0] * (ns - 1) + [-1])
    Delta = -Delta[1:]

    kappa = 1e6
    transform_pars = True

    phi_init, theta_init = arima_transpar(coef_vals, arma, False)
    mod = make_arima(phi_init, theta_init, Delta, kappa)

    def upARIMA(mod, phi, theta):
        p = len(phi)
        q = len(theta)
        mod["phi"] = phi
        mod["theta"] = theta
        r = max(p, q + 1)
        if p > 0:
            mod["T"][:p, 0] = phi
        if r > 1:
            mod["Pn"][:r, :r] = getQ0(phi, theta)
        else:
            mod["Pn"][0, 0] = 1 / (1 - phi[0] ** 2) if p > 0 else 1
        mod["a"][:] = 0
        return mod

    def ml_val_only(p, x, trans):
        x_adj = x.copy()
        par = coef.copy()
        par[mask_arr] = p
        trarma = arima_transpar(par, arma, trans)
        Z = upARIMA(mod, trarma[0], trarma[1])
        if Z is None:
            return np.finfo(np.float64).max
        if ncxreg > 0:
            x_adj = x_adj - np.dot(xreg, par[xreg_idx])
        res = arima_like(
            x_adj, Z["phi"], Z["theta"], Z["delta"], Z["a"], Z["P"], Z["Pn"], 0, False
        )
        if res[2] == 0.0:
            return math.inf
        s2 = res[0] / res[2]
        if s2 <= 0:
            return math.nan
        return 0.5 * (math.log(s2) + res[1] / res[2])

    def ml_val_and_grad(p, x, trans):
        x_adj = x.copy()
        par = coef.copy()
        par[mask_arr] = p
        trarma = arima_transpar(par, arma, trans)
        Z = upARIMA(mod, trarma[0], trarma[1])
        if Z is None:
            return np.finfo(np.float64).max, np.zeros(mask_arr.sum())
        if ncxreg > 0:
            x_adj = x_adj - np.dot(xreg, par[xreg_idx])

        phi_exp = Z["phi"]
        theta_exp = Z["theta"]
        loss_val, d_phi_kf, d_theta_kf, d_Pn_flat, d_y_kf = arima_like_grad(
            x_adj,
            phi_exp,
            theta_exp,
            Z["delta"],
            Z["a"].copy(),
            Z["P"].copy(),
            Z["Pn"].copy(),
            0,
        )

        if not np.isfinite(loss_val):
            return loss_val, np.zeros(mask_arr.sum())

        r_dim = max(len(phi_exp), len(theta_exp) + 1)
        rd_dim = len(Z["a"])
        d_Pn_mat = d_Pn_flat.reshape(rd_dim, rd_dim)
        bar_phi_pn, bar_theta_pn = getQ0_vjp(
            phi_exp, theta_exp, d_Pn_mat, r_dim, len(phi_exp)
        )
        bar_phi_total = d_phi_kf + bar_phi_pn
        bar_theta_total = d_theta_kf + bar_theta_pn

        if trans:
            constrained = arima_undopars(par, arma)
        else:
            constrained = par
        bar_arma = transpar_vjp(
            bar_phi_total, bar_theta_total, constrained[:narma], arma
        )

        if trans:
            J = arima_gradtrans(par, arma)
            bar_arma = J[:narma, :narma].T @ bar_arma

        full_grad = np.zeros(len(par))
        full_grad[:narma] = bar_arma
        if ncxreg > 0:
            full_grad[xreg_idx] = -d_y_kf @ xreg

        return loss_val, full_grad[mask_arr]

    rng = np.random.RandomState(42)
    init = coef_vals + rng.randn(n_params) * 0.05

    optim_opts = {"maxiter": 100}

    res_num = _scipy_minimize(
        ml_val_only,
        init[mask_arr],
        args=(y, transform_pars),
        method="BFGS",
        tol=1e-8,
        options=optim_opts,
    )
    res_ana = _scipy_minimize(
        ml_val_and_grad,
        init[mask_arr],
        args=(y, transform_pars),
        method="BFGS",
        jac=True,
        tol=1e-8,
        options=optim_opts,
    )

    param_match = np.allclose(res_num.x, res_ana.x, atol=1e-3)
    val_match = np.allclose(res_num.fun, res_ana.fun, atol=1e-6)

    for _ in range(3):
        _scipy_minimize(
            ml_val_only,
            init[mask_arr],
            args=(y, transform_pars),
            method="BFGS",
            tol=1e-8,
            options=optim_opts,
        )
        _scipy_minimize(
            ml_val_and_grad,
            init[mask_arr],
            args=(y, transform_pars),
            method="BFGS",
            jac=True,
            tol=1e-8,
            options=optim_opts,
        )

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _scipy_minimize(
            ml_val_only,
            init[mask_arr],
            args=(y, transform_pars),
            method="BFGS",
            tol=1e-8,
            options=optim_opts,
        )
    t_num = (time.perf_counter() - t0) / n_reps

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _scipy_minimize(
            ml_val_and_grad,
            init[mask_arr],
            args=(y, transform_pars),
            method="BFGS",
            jac=True,
            tol=1e-8,
            options=optim_opts,
        )
    t_ana = (time.perf_counter() - t0) / n_reps

    n_p = len(init)
    print(f"  {label} (N={len(y)}, params={n_p})")
    print(
        f"    numerical:  {t_num*1000:8.2f} ms  (nit={res_num.nit}, nfev={res_num.nfev})"
    )
    print(
        f"    analytical: {t_ana*1000:8.2f} ms  (nit={res_ana.nit}, nfev={res_ana.nfev})"
    )
    print(f"    speedup:    {t_num/t_ana:.2f}x")
    print(f"    params match: {param_match}  val match: {val_match}")
    print()


# ---------------------------------------------------------------------------
# End-to-end benchmark: arima(method='CSS-ML') with and without
# analytical gradients. Uses mock.patch to force numerical gradients
# while keeping BFGS as the optimizer for a fair comparison.
# ---------------------------------------------------------------------------
def bench_e2e_case(label, y, order, seasonal=None, include_mean=False, n_reps=50):
    if seasonal is None:
        seasonal = {"order": (0, 0, 0), "period": 1}

    kw = dict(
        order=order,
        seasonal=seasonal,
        include_mean=include_mean,
        method="CSS-ML",
        optim_method="BFGS",
    )

    # Warmup
    for _ in range(3):
        arima(y, **kw)
    with patch("statsforecast.arima.minimize", _minimize_force_numerical):
        for _ in range(3):
            arima(y, **kw)

    # BFGS with analytical gradients (new)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        res_ana = arima(y, **kw)
    t_ana = (time.perf_counter() - t0) / n_reps

    # BFGS with numerical gradients (old, via monkeypatch)
    with patch("statsforecast.arima.minimize", _minimize_force_numerical):
        t0 = time.perf_counter()
        for _ in range(n_reps):
            res_num = arima(y, **kw)
        t_num = (time.perf_counter() - t0) / n_reps

    loglik_ana = res_ana["loglik"]
    loglik_num = res_num["loglik"]
    loglik_match = abs(loglik_ana - loglik_num) < 5.0

    print(f"  {label} (N={len(y)})")
    print(f"    numerical:  {t_num*1000:8.2f} ms  (loglik={loglik_num:.4f})")
    print(f"    analytical: {t_ana*1000:8.2f} ms  (loglik={loglik_ana:.4f})")
    print(f"    speedup:    {t_num/t_ana:.2f}x")
    print(f"    loglik close: {loglik_match}")
    print()


def main():
    print("=" * 70)
    print("  ARIMA CSS Optimization: Analytical vs Numerical Gradients")
    print("=" * 70)
    print()

    bench_css_case("ARIMA(1,1,1)", ap.astype(np.float64), order=(1, 1, 1))
    bench_css_case("ARIMA(2,1,2)", ap.astype(np.float64), order=(2, 1, 2))
    bench_css_case("ARIMA(3,1,3)", ap.astype(np.float64), order=(3, 1, 3))

    np.random.seed(42)
    y_stationary = np.random.randn(500).astype(np.float64) + 5.0
    bench_css_case(
        "ARIMA(1,0,1)+mean, n=500", y_stationary, order=(1, 0, 1), include_mean=True
    )

    np.random.seed(42)
    y_seasonal = np.random.randn(300).astype(np.float64)
    bench_css_case(
        "SARIMA(1,0,1)(1,0,1)[12]",
        y_seasonal,
        order=(1, 0, 1),
        seasonal={"order": (1, 0, 1), "period": 12},
    )

    bench_css_case(
        "SARIMA(2,1,1)(1,1,1)[12]",
        ap.astype(np.float64),
        order=(2, 1, 1),
        seasonal={"order": (1, 1, 1), "period": 12},
    )

    np.random.seed(42)
    y_long = np.cumsum(np.random.randn(2000)).astype(np.float64)
    bench_css_case("ARIMA(2,1,2), n=2000", y_long, order=(2, 1, 2))

    np.random.seed(42)
    y_vlong = np.cumsum(np.random.randn(5000)).astype(np.float64)
    bench_css_case("ARIMA(3,1,3), n=5000", y_vlong, order=(3, 1, 3), n_reps=20)

    print()
    print("=" * 70)
    print("  ARIMA ML Optimization: Analytical vs Numerical Gradients")
    print("=" * 70)
    print()

    bench_ml_case("ARIMA(1,1,1)", ap.astype(np.float64), order=(1, 1, 1))
    bench_ml_case("ARIMA(2,1,2)", ap.astype(np.float64), order=(2, 1, 2))
    bench_ml_case("ARIMA(3,1,3)", ap.astype(np.float64), order=(3, 1, 3))

    np.random.seed(42)
    y_stationary = np.random.randn(500).astype(np.float64) + 5.0
    bench_ml_case(
        "ARIMA(1,0,1)+mean, n=500", y_stationary, order=(1, 0, 1), include_mean=True
    )

    np.random.seed(42)
    y_seasonal = np.random.randn(300).astype(np.float64)
    bench_ml_case(
        "SARIMA(1,0,1)(1,0,1)[12]",
        y_seasonal,
        order=(1, 0, 1),
        seasonal={"order": (1, 0, 1), "period": 12},
    )

    bench_ml_case(
        "SARIMA(2,1,1)(1,1,1)[12]",
        ap.astype(np.float64),
        order=(2, 1, 1),
        seasonal={"order": (1, 1, 1), "period": 12},
    )

    np.random.seed(42)
    y_long = np.cumsum(np.random.randn(2000)).astype(np.float64)
    bench_ml_case("ARIMA(2,1,2), n=2000", y_long, order=(2, 1, 2))

    np.random.seed(42)
    y_vlong = np.cumsum(np.random.randn(5000)).astype(np.float64)
    bench_ml_case("ARIMA(3,1,3), n=5000", y_vlong, order=(3, 1, 3), n_reps=20)

    print()
    print("=" * 70)
    print("  End-to-End: arima(method='CSS-ML') BFGS analytical vs numerical")
    print("=" * 70)
    print()

    bench_e2e_case("ARIMA(1,1,1)", ap.astype(np.float64), order=(1, 1, 1))
    bench_e2e_case("ARIMA(2,1,2)", ap.astype(np.float64), order=(2, 1, 2))

    np.random.seed(42)
    y_seasonal = np.random.randn(300).astype(np.float64)
    bench_e2e_case(
        "SARIMA(1,0,1)(1,0,1)[12]",
        y_seasonal,
        order=(1, 0, 1),
        seasonal={"order": (1, 0, 1), "period": 12},
    )

    bench_e2e_case(
        "SARIMA(2,1,1)(1,1,1)[12]",
        ap.astype(np.float64),
        order=(2, 1, 1),
        seasonal={"order": (1, 1, 1), "period": 12},
    )

    np.random.seed(42)
    y_long = np.cumsum(np.random.randn(2000)).astype(np.float64)
    bench_e2e_case("ARIMA(2,1,2), n=2000", y_long, order=(2, 1, 2))

    np.random.seed(42)
    y_vlong = np.cumsum(np.random.randn(5000)).astype(np.float64)
    bench_e2e_case("ARIMA(3,1,3), n=5000", y_vlong, order=(3, 1, 3), n_reps=20)


if __name__ == "__main__":
    main()
