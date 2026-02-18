"""
Benchmark: Old Numba vs New C++ implementations
================================================

Compares output equivalence and execution speed for all functions
migrated from numba @njit to C++ (pybind11+Eigen).

Usage:
    .venv/bin/python experiments/numba_removal/benchmark.py
"""

import sys
import time
import timeit
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Series generation (utilsforecast)
# ---------------------------------------------------------------------------
from utilsforecast.data import generate_series

# ---------------------------------------------------------------------------
# NEW (C++) implementations
# ---------------------------------------------------------------------------
from statsforecast._lib import ces as _ces
from statsforecast._lib import garch as _garch
from statsforecast._lib import mfles as _mfles
from statsforecast._lib import ses as _ses
from statsforecast._lib import tbats as _tbats

# ---------------------------------------------------------------------------
# OLD (numba) implementations
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _old_numba.ces_old import (
    cescalc as old_cescalc,
    cesforecast as old_cesforecast,
    switch_ces as old_switch_ces,
)
from _old_numba.garch_old import (
    garch_loglik as old_garch_loglik,
    garch_sigma2 as old_garch_sigma2,
)
from _old_numba.mfles_old import (
    get_basis as old_get_basis,
    siegel_repeated_medians as old_siegel_repeated_medians,
)
from _old_numba.ses_old import (
    _expand_fitted_demand as old_expand_fitted_demand,
    _expand_fitted_intervals as old_expand_fitted_intervals,
    _ses_forecast as old_ses_forecast,
    _ses_sse as old_ses_sse,
)
from _old_numba.tbats_old import (
    calcTBATSFaster as old_calcTBATSFaster,
    makeTBATSFMatrix as old_makeTBATSFMatrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Time-series generation helpers
# ---------------------------------------------------------------------------
SEED = 42
N = 500  # default series length


def _build_series_cache():
    """Generate all benchmark series using utilsforecast's generate_series."""
    rng = np.random.RandomState(SEED)

    # Seasonal + trend series from utilsforecast
    n_generated = 100
    df = generate_series(
        n_series=n_generated,
        freq="D",
        min_length=N,
        max_length=N,
        with_trend=True,
        seed=SEED,
    )
    uids = df["unique_id"].unique()

    series = {}
    series["ets_seasonal"] = df.loc[df["unique_id"] == uids[0], "y"].values.astype(np.float64)
    series["seasonal"] = df.loc[df["unique_id"] == uids[1], "y"].values.astype(np.float64)
    series["ets_trend"] = df.loc[df["unique_id"] == uids[2], "y"].values.astype(np.float64)

    # Random walk: cumulative sum of noise
    series["random_walk"] = np.cumsum(rng.randn(N)).astype(np.float64)

    # Intermittent demand: sparse positive values
    base = rng.exponential(5.0, size=N)
    mask = rng.random(N) > 0.7  # ~30% non-zero
    series["intermittent"] = (base * mask).astype(np.float64)

    # GARCH-like: heteroskedastic noise
    y_garch = np.zeros(N)
    sigma2 = np.zeros(N)
    sigma2[0] = 1.0
    for t in range(1, N):
        sigma2[t] = 0.1 + 0.2 * y_garch[t - 1] ** 2 + 0.7 * sigma2[t - 1]
        y_garch[t] = rng.randn() * np.sqrt(sigma2[t])
    series["garch"] = y_garch.astype(np.float64)

    return series


_series_cache = _build_series_cache()


def gen(name: str) -> np.ndarray:
    return _series_cache[name].copy()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
REPEAT = 5
NUMBER = 20


def bench(fn, *args, number=NUMBER, repeat=REPEAT, **kwargs):
    """Return (min_time_ms, result_of_one_call)."""
    result = fn(*args, **kwargs)
    times = timeit.repeat(lambda: fn(*args, **kwargs), number=number, repeat=repeat)
    min_ms = min(times) / number * 1000
    return min_ms, result


def warmup_numba(fn, *args, **kwargs):
    """Call a numba function once to trigger JIT compilation."""
    fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Result table
# ---------------------------------------------------------------------------
results = []


def record(model, series_type, old_ms, new_ms, match, notes=""):
    speedup = old_ms / new_ms if new_ms > 0 else float("inf")
    results.append({
        "model": model,
        "series": series_type,
        "old_ms": old_ms,
        "new_ms": new_ms,
        "speedup": speedup,
        "match": match,
        "notes": notes,
    })


def check_close(a, b, rtol=1e-4, atol=1e-6, label=""):
    """Check np.allclose and return (bool, info_string)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return True, "all NaN"
    max_diff = np.max(np.abs(a[mask] - b[mask]))
    ok = np.allclose(a[mask], b[mask], rtol=rtol, atol=atol)
    return ok, f"max_diff={max_diff:.2e}"


# ===================================================================
# Benchmarks
# ===================================================================

def benchmark_ses_sse():
    """SES SSE: _ses_sse"""
    for stype in ("ets_seasonal", "ets_trend", "random_walk"):
        y = gen(stype)
        alpha = 0.3

        # warmup numba
        warmup_numba(old_ses_sse, alpha, y)

        old_ms, old_res = bench(old_ses_sse, alpha, y)
        new_ms, new_res = bench(_ses.ses_sse, alpha, y)
        ok, info = check_close(old_res, new_res)
        record("ses_sse", stype, old_ms, new_ms, ok, info)


def benchmark_ses_forecast():
    """SES forecast: _ses_forecast"""
    for stype in ("ets_seasonal", "ets_trend", "random_walk"):
        y = gen(stype)
        alpha = 0.3

        warmup_numba(old_ses_forecast, y, alpha)

        old_ms, (old_fc, old_fitted) = bench(old_ses_forecast, y, alpha)
        new_ms, (new_fc, new_fitted) = bench(_ses.ses_forecast, y, alpha)
        ok1, i1 = check_close(old_fc, new_fc, label="forecast")
        ok2, i2 = check_close(old_fitted, new_fitted, label="fitted")
        record("ses_forecast", stype, old_ms, new_ms, ok1 and ok2, f"fc:{i1} fitted:{i2}")


def benchmark_expand_fitted():
    """Expand fitted demand/intervals for intermittent models"""
    y = gen("intermittent")
    # Ensure some positive values for demand
    y_pos = np.abs(y) + 0.1

    # Create simple fitted arrays (simulate SES output on demand component)
    demand = y_pos[y_pos > 0]
    n_demand = len(demand)
    fitted_demand = np.empty(n_demand + 1, dtype=y.dtype)
    fitted_demand[0] = demand[0]
    for i in range(1, n_demand + 1):
        idx = min(i, n_demand - 1)
        fitted_demand[i] = 0.3 * demand[idx] + 0.7 * fitted_demand[i - 1]

    # intervals
    intervals = np.diff(np.where(y_pos > 0)[0] + 1, prepend=0).astype(y.dtype)
    n_int = len(intervals)
    fitted_int = np.empty(n_int + 1, dtype=y.dtype)
    fitted_int[0] = intervals[0] if n_int > 0 else 1.0
    for i in range(1, n_int + 1):
        idx = min(i, n_int - 1)
        fitted_int[i] = 0.3 * intervals[idx] + 0.7 * fitted_int[i - 1]

    # Demand expand
    warmup_numba(old_expand_fitted_demand, fitted_demand, y_pos)
    old_ms, old_res = bench(old_expand_fitted_demand, fitted_demand, y_pos)
    new_ms, new_res = bench(_ses.expand_fitted_demand, fitted_demand, y_pos)
    ok, info = check_close(old_res, new_res)
    record("expand_demand", "intermittent", old_ms, new_ms, ok, info)

    # Interval expand
    warmup_numba(old_expand_fitted_intervals, fitted_int, y_pos)
    old_ms, old_res = bench(old_expand_fitted_intervals, fitted_int, y_pos)
    new_ms, new_res = bench(_ses.expand_fitted_intervals, fitted_int, y_pos)
    ok, info = check_close(old_res, new_res)
    record("expand_intervals", "intermittent", old_ms, new_ms, ok, info)


def benchmark_garch():
    """GARCH: sigma2 and loglik"""
    y = gen("garch")
    p, q = 1, 1
    coeff = np.array([0.1, 0.2, 0.7])

    warmup_numba(old_garch_sigma2, coeff, y, p, q)
    warmup_numba(old_garch_loglik, coeff, y, p, q)

    # sigma2
    old_ms, old_s2 = bench(old_garch_sigma2, coeff, y, p, q)
    new_ms, new_s2 = bench(_garch.compute_sigma2, coeff, y, p, q)
    ok, info = check_close(old_s2, new_s2)
    record("garch_sigma2", "garch", old_ms, new_ms, ok, info)

    # loglik
    old_ms, old_ll = bench(old_garch_loglik, coeff, y, p, q)
    new_ms, new_ll = bench(_garch.loglik, coeff, y, p, q)
    ok, info = check_close(old_ll, new_ll)
    record("garch_loglik", "garch", old_ms, new_ms, ok, info)


def benchmark_ces():
    """CES: cescalc and cesforecast"""
    from statsmodels.tsa.seasonal import seasonal_decompose

    for stype in ("ets_seasonal", "seasonal"):
        y_orig = gen(stype)
        m = 12
        n = len(y_orig)
        alpha_0, alpha_1 = 1.5, 1.0
        beta_0, beta_1 = 0.9, 0.0
        season = 2  # PARTIAL
        nmse = 3

        # Init states (matching old initstate for "P")
        lags = m
        init_states = np.zeros((lags, 3), dtype=np.float32)
        init_states[:lags, 0] = np.mean(y_orig[:lags]).astype(np.float32)
        init_states[:lags, 1] = (init_states[:lags, 0] / 1.1).astype(np.float32)
        try:
            init_states[:lags, 2] = seasonal_decompose(
                y_orig, period=lags
            ).seasonal[:lags].astype(np.float32)
        except Exception:
            init_states[:lags, 2] = 0.0

        # --- cescalc ---
        # Old numba
        def run_old_cescalc():
            s = np.zeros((m + n, 3), dtype=np.float32)
            s[:m] = init_states.copy()
            y_c = y_orig.copy()
            e = np.zeros(n)
            amse = np.zeros(30)
            old_cescalc(y_c, s, m, season, alpha_0, alpha_1, beta_0, beta_1,
                        e, amse, nmse, 1)
            return s, e, amse

        # New C++
        def run_new_cescalc():
            s = np.zeros((m + n, 3), dtype=np.float32)
            s[:m] = init_states.copy()
            y_c = y_orig.copy()
            e = np.zeros(n)
            amse = np.zeros(30)
            _ces.cescalc(y_c, s, m, season, float(alpha_0), float(alpha_1),
                         float(beta_0), float(beta_1), e, amse, nmse, 1)
            return s, e, amse

        warmup_numba(old_cescalc,
                     y_orig.copy(),
                     np.zeros((m + n, 3), dtype=np.float32),
                     m, season, alpha_0, alpha_1, beta_0, beta_1,
                     np.zeros(n), np.zeros(30), nmse, 1)

        old_ms, (old_s, old_e, old_amse) = bench(run_old_cescalc, number=5, repeat=3)
        new_ms, (new_s, new_e, new_amse) = bench(run_new_cescalc, number=5, repeat=3)

        ok1, i1 = check_close(old_s, new_s, rtol=1e-3, atol=1e-2)
        ok2, i2 = check_close(old_e, new_e, rtol=1e-3, atol=1e-2)
        record("cescalc", stype, old_ms, new_ms, ok1 and ok2,
               f"states:{i1} errors:{i2}")

        # --- cesforecast ---
        # Run cescalc first to get states
        states_old = np.zeros((m + n, 3), dtype=np.float32)
        states_old[:m] = init_states.copy()
        y_c = y_orig.copy()
        e_tmp = np.zeros(n)
        amse_tmp = np.zeros(30)
        old_cescalc(y_c, states_old, m, season, alpha_0, alpha_1, beta_0, beta_1,
                    e_tmp, amse_tmp, nmse, 1)

        h = 12
        def run_old_cesfcst():
            f = np.zeros(h, dtype=np.float32)
            old_cesforecast(states_old.copy(), n, m, season, f, h,
                            alpha_0, alpha_1, beta_0, beta_1)
            return f

        def run_new_cesfcst():
            f_f64 = np.zeros(h, dtype=np.float64)
            _ces.forecast(
                np.asarray(states_old, dtype=np.float32),
                n, m, "P", f_f64, h,
                float(alpha_0), float(alpha_1), float(beta_0), float(beta_1),
            )
            return f_f64

        warmup_numba(old_cesforecast,
                     states_old.copy(), n, m, season, np.zeros(h, dtype=np.float32), h,
                     alpha_0, alpha_1, beta_0, beta_1)

        old_ms, old_f = bench(run_old_cesfcst, number=10, repeat=3)
        new_ms, new_f = bench(run_new_cesfcst, number=10, repeat=3)
        ok, info = check_close(old_f, new_f, rtol=1e-3, atol=1e-2)
        record("cesforecast", stype, old_ms, new_ms, ok, info)


def benchmark_tbats():
    """TBATS: makeFMatrix and calcFaster"""
    y = gen("ets_seasonal")
    n = len(y)

    # Parameters for TBATS with one seasonal period (m=12, k=3)
    seasonal_periods = np.array([12.0])
    k_vector = np.array([3])
    tau = int(2 * np.sum(k_vector))  # 6
    phi = 0.99
    alpha = 0.1
    beta = 0.05
    ar_coeffs = np.array([0.3, 0.1])
    ma_coeffs = np.array([0.2])
    gamma_bold = np.random.RandomState(42).randn(1, tau)

    # --- makeFMatrix ---
    warmup_numba(old_makeTBATSFMatrix,
                 phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
                 gamma_bold, seasonal_periods, k_vector)

    old_ms, old_F = bench(
        old_makeTBATSFMatrix, phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
        gamma_bold, seasonal_periods, k_vector
    )
    new_ms, new_F = bench(
        _tbats.makeFMatrix, phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
        gamma_bold, seasonal_periods, k_vector
    )
    ok, info = check_close(old_F, new_F)
    record("makeFMatrix", "ets_seasonal", old_ms, new_ms, ok, info)

    # --- calcFaster ---
    # Build w_transpose and g matrices
    from statsforecast.tbats import makeTBATSWMatrix, makeTBATSGMatrix

    p, q = len(ar_coeffs), len(ma_coeffs)
    w = makeTBATSWMatrix(phi, k_vector, ar_coeffs, ma_coeffs, tau)
    g, gb = makeTBATSGMatrix(k_vector, alpha, 1, beta,
                              np.array([0.01]*len(k_vector)),
                              np.array([0.01]*len(k_vector)),
                              p, q, tau)
    F_mat = np.asarray(new_F, dtype=np.float64)
    x0 = np.zeros(F_mat.shape[0])
    x0[0] = np.mean(y[:12])

    w_t = np.asarray(w, dtype=np.float64)
    g_mat = np.asarray(g, dtype=np.float64)
    y_trans = y.copy()

    warmup_numba(old_calcTBATSFaster, y_trans, w_t, g_mat, F_mat, x0)

    old_ms, (old_yhat, old_e, old_x) = bench(
        old_calcTBATSFaster, y_trans, w_t, g_mat, F_mat, x0
    )
    new_ms, (new_yhat, new_e, new_x) = bench(
        _tbats.calcFaster, y_trans, w_t.ravel(), g_mat, F_mat, x0
    )
    ok1, i1 = check_close(old_yhat, new_yhat)
    ok2, i2 = check_close(old_e, new_e)
    record("calcFaster", "ets_seasonal", old_ms, new_ms, ok1 and ok2,
           f"yhat:{i1} err:{i2}")


def benchmark_mfles():
    """MFLES: get_basis and siegel_repeated_medians"""
    for stype in ("ets_seasonal", "ets_trend", "seasonal"):
        y = gen(stype)

        # --- get_basis ---
        n_cp = 3
        warmup_numba(old_get_basis, y, n_cp)

        old_ms, old_basis = bench(old_get_basis, y, n_cp)
        # C++ expects column vector (m, 1) and explicit decay/gradient_strategy args
        new_ms, new_basis = bench(_mfles.get_basis, y.reshape(-1, 1), n_cp, -1.0, 0)
        ok, info = check_close(old_basis, new_basis)
        record("get_basis", stype, old_ms, new_ms, ok, info)

        # --- siegel_repeated_medians ---
        x = np.arange(len(y), dtype=np.float64)
        warmup_numba(old_siegel_repeated_medians, x, y)

        old_ms, old_med = bench(old_siegel_repeated_medians, x, y, number=3, repeat=3)
        # C++ expects column vectors (m, 1)
        new_ms, new_med = bench(
            _mfles.siegel_repeated_medians, x.reshape(-1, 1), y.reshape(-1, 1),
            number=3, repeat=3,
        )
        ok, info = check_close(old_med, new_med)
        record("siegel_medians", stype, old_ms, new_ms, ok, info)


# ===================================================================
# Fitted Values Benchmarks
# ===================================================================

def benchmark_ses_fitted():
    """Full SES pipeline: optimize alpha + compute fitted values."""
    from scipy.optimize import minimize_scalar

    for stype in ("ets_seasonal", "ets_trend", "random_walk"):
        y = gen(stype)

        # --- Old numba path ---
        def run_old():
            alpha = minimize_scalar(
                fun=old_ses_sse, bounds=(0.01, 0.99), args=(y,),
            ).x
            forecast, fitted = old_ses_forecast(y, alpha)
            return forecast, fitted, alpha

        # --- New C++ path ---
        def run_new():
            alpha = minimize_scalar(
                fun=_ses.ses_sse, bounds=(0.01, 0.99), args=(y,),
            ).x
            forecast, fitted = _ses.ses_forecast(y, alpha)
            return forecast, fitted, alpha

        # warmup numba
        warmup_numba(old_ses_sse, 0.3, y)
        warmup_numba(old_ses_forecast, y, 0.3)

        old_ms, (old_fc, old_fitted, old_alpha) = bench(run_old, number=5, repeat=3)
        new_ms, (new_fc, new_fitted, new_alpha) = bench(run_new, number=5, repeat=3)

        ok1, i1 = check_close(old_alpha, new_alpha, label="alpha")
        ok2, i2 = check_close(old_fitted, new_fitted, label="fitted")
        ok3, i3 = check_close(old_fc, new_fc, label="forecast")
        record("ses_fitted", stype, old_ms, new_ms, ok1 and ok2 and ok3,
               f"alpha:{i1} fitted:{i2}")


def benchmark_croston_fitted():
    """Croston Classic full pipeline with fitted values."""
    from statsforecast.utils import _ensure_float

    y = gen("intermittent")
    y = _ensure_float(y)

    def _demand(x):
        return x[x > 0]

    def _intervals(x):
        nonzero_idxs = np.where(x != 0)[0]
        return np.diff(nonzero_idxs + 1, prepend=0).astype(x.dtype)

    # Skip if no demand
    yd = _demand(y)
    if not yd.size:
        return

    # --- Old numba path ---
    def run_old():
        ydp, ydf = old_ses_forecast(yd, 0.1)
        yi = _intervals(y)
        yip, yif = old_ses_forecast(yi, 0.1)
        mean = ydp / yip if yip != 0.0 else ydp
        ydf_exp = old_expand_fitted_demand(np.append(ydf, ydp), y)
        yif_exp = old_expand_fitted_intervals(np.append(yif, yip), y)
        fitted = ydf_exp / yif_exp
        return mean, fitted

    # --- New C++ path ---
    def run_new():
        ydp, ydf = _ses.ses_forecast(yd, 0.1)
        yi = _intervals(y)
        yip, yif = _ses.ses_forecast(yi, 0.1)
        mean = ydp / yip if yip != 0.0 else ydp
        ydf_exp = _ses.expand_fitted_demand(np.append(ydf, ydp), y)
        yif_exp = _ses.expand_fitted_intervals(np.append(yif, yip), y)
        fitted = ydf_exp / yif_exp
        return mean, fitted

    # warmup
    warmup_numba(old_ses_forecast, yd, 0.1)
    warmup_numba(old_expand_fitted_demand,
                 np.append(old_ses_forecast(yd, 0.1)[1],
                           old_ses_forecast(yd, 0.1)[0]), y)

    old_ms, (old_mean, old_fitted) = bench(run_old, number=10, repeat=3)
    new_ms, (new_mean, new_fitted) = bench(run_new, number=10, repeat=3)

    ok1, i1 = check_close(old_mean, new_mean)
    ok2, i2 = check_close(old_fitted, new_fitted)
    record("croston_fitted", "intermittent", old_ms, new_ms, ok1 and ok2,
           f"mean:{i1} fitted:{i2}")


def benchmark_garch_fitted():
    """Full GARCH model fit + forecast with fitted values."""
    from scipy.optimize import minimize

    y = gen("garch")
    p, q = 1, 1

    def garch_cons_old(x0):
        return 1 - (x0[1:].sum())

    def garch_cons_new(x0):
        return 1 - (x0[1:].sum())

    # --- Old numba path ---
    def run_old():
        x0 = np.full(p + q + 1, 0.1)
        bnds = ((1e-8, None),) * len(x0)
        cons = {"type": "ineq", "fun": garch_cons_old}
        opt = minimize(
            old_garch_loglik, x0, args=(y, p, q),
            method="SLSQP", bounds=bnds, constraints=cons,
        )
        coeff = opt.x
        sigma2 = old_garch_sigma2(coeff, y, p, q)
        np.random.seed(1)
        fitted = np.full(len(y), np.nan)
        for k in range(p, len(y)):
            error = np.random.randn()
            fitted[k] = error * np.sqrt(sigma2[k])
        return coeff, sigma2, fitted

    # --- New C++ path ---
    def run_new():
        x0 = np.full(p + q + 1, 0.1)
        bnds = ((1e-8, None),) * len(x0)
        cons = {"type": "ineq", "fun": garch_cons_new}
        def obj(params):
            return _garch.loglik(params, y, p, q)
        opt = minimize(
            obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
        )
        coeff = opt.x
        sigma2 = _garch.compute_sigma2(coeff, y, p, q)
        np.random.seed(1)
        fitted = np.full(len(y), np.nan)
        for k in range(p, len(y)):
            error = np.random.randn()
            fitted[k] = error * np.sqrt(sigma2[k])
        return coeff, sigma2, fitted

    warmup_numba(old_garch_loglik, np.full(3, 0.1), y, p, q)
    warmup_numba(old_garch_sigma2, np.full(3, 0.1), y, p, q)

    old_ms, (old_coeff, old_s2, old_fitted) = bench(run_old, number=1, repeat=3)
    new_ms, (new_coeff, new_s2, new_fitted) = bench(run_new, number=1, repeat=3)

    ok1, i1 = check_close(old_coeff, new_coeff, rtol=1e-3)
    ok2, i2 = check_close(old_s2, new_s2, rtol=1e-3)
    ok3, i3 = check_close(old_fitted, new_fitted, rtol=1e-3)
    record("garch_fitted", "garch", old_ms, new_ms, ok1 and ok2 and ok3,
           f"coeff:{i1} sigma2:{i2} fitted:{i3}")


def benchmark_ces_fitted():
    """Full CES model fit producing fitted values."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    y_orig = gen("ets_seasonal")
    m = 12
    n = len(y_orig)
    seasontype = "P"
    nmse = 3

    # Init states
    lags = m
    init_state = np.zeros((lags, 3), dtype=np.float32)
    init_state[:lags, 0] = np.mean(y_orig[:lags]).astype(np.float32)
    init_state[:lags, 1] = (init_state[:lags, 0] / 1.1).astype(np.float32)
    try:
        init_state[:lags, 2] = seasonal_decompose(
            y_orig, period=lags
        ).seasonal[:lags].astype(np.float32)
    except Exception:
        init_state[:lags, 2] = 0.0

    n_components = init_state.shape[1]
    alpha_0, alpha_1 = 1.5, 1.0
    beta_0, beta_1 = 0.9, 0.0
    season = 2  # PARTIAL

    # --- Old numba path: cescalc to get fitted = y - e ---
    def run_old():
        s = np.zeros((m + n, n_components), dtype=np.float32)
        s[:m] = init_state.copy()
        y_c = y_orig.copy()
        e = np.zeros(n)
        amse = np.zeros(30)
        lik = old_cescalc(y_c, s, m, season, alpha_0, alpha_1, beta_0, beta_1,
                          e, amse, nmse, 1)
        fitted = y_orig - e
        return fitted, s, e, lik

    # --- New C++ path ---
    def run_new():
        s = np.zeros((m + n, n_components), dtype=np.float32)
        s[:m] = init_state.copy()
        y_c = y_orig.copy()
        e = np.zeros(n)
        amse = np.zeros(30)
        lik = _ces.cescalc(y_c, s, m, season, float(alpha_0), float(alpha_1),
                           float(beta_0), float(beta_1), e, amse, nmse, 1)
        fitted = y_orig - e
        return fitted, s, e, lik

    warmup_numba(old_cescalc, y_orig.copy(),
                 np.zeros((m + n, n_components), dtype=np.float32),
                 m, season, alpha_0, alpha_1, beta_0, beta_1,
                 np.zeros(n), np.zeros(30), nmse, 1)

    old_ms, (old_fitted, old_s, old_e, old_lik) = bench(run_old, number=5, repeat=3)
    new_ms, (new_fitted, new_s, new_e, new_lik) = bench(run_new, number=5, repeat=3)

    ok1, i1 = check_close(old_fitted, new_fitted, rtol=1e-3, atol=1e-2)
    ok2, i2 = check_close(old_lik, new_lik, rtol=1e-3)
    record("ces_fitted", "ets_seasonal", old_ms, new_ms, ok1 and ok2,
           f"fitted:{i1} lik:{i2}")


def benchmark_tbats_fitted():
    """TBATS calcFaster produces fitted values (yhat) as a byproduct."""
    y = gen("ets_seasonal")
    n = len(y)

    seasonal_periods = np.array([12.0])
    k_vector = np.array([3])
    tau = int(2 * np.sum(k_vector))
    phi = 0.99
    alpha = 0.1
    beta = 0.05
    ar_coeffs = np.array([0.3, 0.1])
    ma_coeffs = np.array([0.2])
    gamma_bold = np.random.RandomState(42).randn(1, tau)

    from statsforecast.tbats import makeTBATSWMatrix, makeTBATSGMatrix

    F_mat = np.asarray(
        _tbats.makeFMatrix(phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
                           gamma_bold, seasonal_periods, k_vector),
        dtype=np.float64,
    )
    p, q = len(ar_coeffs), len(ma_coeffs)
    w = makeTBATSWMatrix(phi, k_vector, ar_coeffs, ma_coeffs, tau)
    g, gb = makeTBATSGMatrix(k_vector, alpha, 1, beta,
                              np.array([0.01]*len(k_vector)),
                              np.array([0.01]*len(k_vector)),
                              p, q, tau)
    x0 = np.zeros(F_mat.shape[0])
    x0[0] = np.mean(y[:12])
    w_t = np.asarray(w, dtype=np.float64)
    g_mat = np.asarray(g, dtype=np.float64)

    # Old numba: calcTBATSFaster returns (yhat, e, x) — yhat IS the fitted
    warmup_numba(old_calcTBATSFaster, y, w_t, g_mat, F_mat, x0)

    def run_old():
        yhat, e, x = old_calcTBATSFaster(y, w_t, g_mat, F_mat, x0)
        return yhat.ravel(), e.ravel()

    def run_new():
        yhat, e, x = _tbats.calcFaster(y, w_t.ravel(), g_mat, F_mat, x0)
        return yhat.ravel(), e.ravel()

    old_ms, (old_yhat, old_e) = bench(run_old)
    new_ms, (new_yhat, new_e) = bench(run_new)

    # yhat = fitted values, e = residuals
    ok1, i1 = check_close(old_yhat, new_yhat)
    ok2, i2 = check_close(old_e, new_e)
    record("tbats_fitted", "ets_seasonal", old_ms, new_ms, ok1 and ok2,
           f"yhat(fitted):{i1} residuals:{i2}")


# ===================================================================
# Prediction Intervals Benchmarks
# ===================================================================

def benchmark_ses_pi():
    """SES prediction intervals (analytical formula, uses sigma from fitted)."""
    from scipy.optimize import minimize_scalar
    from statsforecast.utils import _calculate_intervals

    for stype in ("ets_seasonal", "random_walk"):
        y = gen(stype)
        h = 12
        level = [80, 95]

        # --- Old numba path ---
        def run_old():
            alpha = minimize_scalar(
                fun=old_ses_sse, bounds=(0.01, 0.99), args=(y,),
            ).x
            forecast, fitted = old_ses_forecast(y, alpha)
            residuals = y - fitted
            sigma = np.nanstd(residuals, ddof=1)
            steps = np.arange(1, h + 1)
            sigmah = sigma * np.sqrt(1 + (steps - 1) * alpha**2)
            mean = np.full(h, forecast)
            res = {"mean": mean}
            pi = _calculate_intervals(res, level, h, sigmah)
            return {**res, **pi}, alpha, fitted

        # --- New C++ path ---
        def run_new():
            alpha = minimize_scalar(
                fun=_ses.ses_sse, bounds=(0.01, 0.99), args=(y,),
            ).x
            forecast, fitted = _ses.ses_forecast(y, alpha)
            residuals = y - fitted
            sigma = np.nanstd(residuals, ddof=1)
            steps = np.arange(1, h + 1)
            sigmah = sigma * np.sqrt(1 + (steps - 1) * alpha**2)
            mean = np.full(h, forecast)
            res = {"mean": mean}
            pi = _calculate_intervals(res, level, h, sigmah)
            return {**res, **pi}, alpha, fitted

        warmup_numba(old_ses_sse, 0.3, y)
        warmup_numba(old_ses_forecast, y, 0.3)

        old_ms, (old_res, old_alpha, old_fitted) = bench(run_old, number=5, repeat=3)
        new_ms, (new_res, new_alpha, new_fitted) = bench(run_new, number=5, repeat=3)

        ok_mean, im = check_close(old_res["mean"], new_res["mean"])
        ok_lo = True
        ok_hi = True
        for lv in level:
            o1, _ = check_close(old_res[f"lo-{lv}"], new_res[f"lo-{lv}"])
            o2, _ = check_close(old_res[f"hi-{lv}"], new_res[f"hi-{lv}"])
            ok_lo = ok_lo and o1
            ok_hi = ok_hi and o2
        ok_fitted, i_fitted = check_close(old_fitted, new_fitted)
        record("ses_pi", stype, old_ms, new_ms,
               ok_mean and ok_lo and ok_hi and ok_fitted,
               f"mean:{im} fitted:ok={ok_fitted} pi_lo:ok={ok_lo} pi_hi:ok={ok_hi}")


def benchmark_ces_pi():
    """CES prediction intervals via Monte Carlo simulation (5000 paths)."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsforecast.ces import cesforecast as new_cesforecast, switch_ces

    y_orig = gen("ets_seasonal")
    m = 12
    n = len(y_orig)
    seasontype = "N"  # non-seasonal for speed (avoids seasonal_decompose in sim)
    nmse = 3
    h = 12
    level = [80, 95]
    nsim = 500  # reduced for benchmark speed

    # Init states for non-seasonal
    init_state = np.zeros((1, 2), dtype=np.float32)
    idx = min(max(10, m), n)
    mean_ = np.mean(y_orig[:idx])
    init_state[0, 0] = mean_
    init_state[0, 1] = mean_ / 1.1

    n_components = init_state.shape[1]
    alpha_0, alpha_1 = 1.5, 1.0
    beta_0, beta_1 = np.nan, np.nan
    season_int = 0  # NONE

    # First fit the model to get states and sigma2
    m_eff = 1  # non-seasonal
    states_fit = np.zeros((m_eff + n, n_components), dtype=np.float32)
    states_fit[:m_eff] = init_state.copy()
    y_c = y_orig.copy()
    e = np.zeros(n)
    amse = np.zeros(30)
    _ces.cescalc(y_c, states_fit, m_eff, season_int,
                 float(alpha_0), float(alpha_1), 0.0, 0.0,
                 e, amse, nmse, 1)
    fitted = y_orig - e
    sigma2 = np.sum(e**2) / (n - n_components - 1)
    sigma = np.sqrt(sigma2)

    model = {
        "states": states_fit,
        "n": n,
        "m": m_eff,
        "seasontype": seasontype,
        "par": {"alpha_0": alpha_0, "alpha_1": alpha_1, "beta_0": 0.0, "beta_1": 0.0},
        "sigma2": sigma2,
        "fitted": fitted,
    }
    season = switch_ces(seasontype)

    # --- Old numba path (Monte Carlo PI) ---
    def run_old_pi():
        np.random.seed(1)
        y_path = np.zeros([nsim, h])
        for k in range(nsim):
            e_noise = np.random.normal(0, sigma, states_fit.shape)
            f = np.zeros(h, dtype=np.float32)
            old_cesforecast(
                (states_fit + e_noise).astype(np.float32),
                n, m_eff, season_int, f, h,
                alpha_0, alpha_1, 0.0, 0.0,
            )
            y_path[k] = f
        lower = np.quantile(y_path, 0.5 - np.array(level) / 200, axis=0)
        upper = np.quantile(y_path, 0.5 + np.array(level) / 200, axis=0)
        return lower, upper

    # --- New C++ path (Monte Carlo PI) ---
    def run_new_pi():
        np.random.seed(1)
        y_path = np.zeros([nsim, h])
        for k in range(nsim):
            e_noise = np.random.normal(0, sigma, states_fit.shape)
            f = np.zeros(h, dtype=np.float64)
            _ces.forecast(
                (states_fit + e_noise).astype(np.float32),
                n, m_eff, seasontype, f, h,
                float(alpha_0), float(alpha_1), 0.0, 0.0,
            )
            y_path[k] = f
        lower = np.quantile(y_path, 0.5 - np.array(level) / 200, axis=0)
        upper = np.quantile(y_path, 0.5 + np.array(level) / 200, axis=0)
        return lower, upper

    warmup_numba(old_cesforecast,
                 states_fit.copy(), n, m_eff, season_int,
                 np.zeros(h, dtype=np.float32), h,
                 alpha_0, alpha_1, 0.0, 0.0)

    old_ms, (old_lo, old_hi) = bench(run_old_pi, number=1, repeat=3)
    new_ms, (new_lo, new_hi) = bench(run_new_pi, number=1, repeat=3)

    # Same seed → same noise → outputs should match closely
    ok_lo, i_lo = check_close(old_lo, new_lo, rtol=1e-2, atol=1e-1)
    ok_hi, i_hi = check_close(old_hi, new_hi, rtol=1e-2, atol=1e-1)
    record("ces_pi", "ets_seasonal", old_ms, new_ms, ok_lo and ok_hi,
           f"nsim={nsim} lo:{i_lo} hi:{i_hi}")


def benchmark_cold_start():
    """Measure numba cold-start (first call) vs C++ for representative functions."""
    import importlib
    import subprocess

    # We measure cold start by launching a subprocess for each function
    # This ensures numba JIT compilation happens fresh each time

    funcs = [
        ("ses_sse", "from _old_numba.ses_old import _ses_sse; import numpy as np; _ses_sse(0.3, np.random.randn(500))"),
        ("garch_sigma2", "from _old_numba.garch_old import garch_sigma2; import numpy as np; garch_sigma2(np.array([0.1, 0.2, 0.7]), np.random.randn(500), 1, 1)"),
        ("get_basis", "from _old_numba.mfles_old import get_basis; import numpy as np; get_basis(np.random.randn(500), 3)"),
    ]

    cpp_funcs = [
        ("ses_sse", "from statsforecast._lib import ses; import numpy as np; ses.ses_sse(0.3, np.random.randn(500))"),
        ("garch_sigma2", "from statsforecast._lib import garch; import numpy as np; garch.compute_sigma2(np.array([0.1, 0.2, 0.7]), np.random.randn(500), 1, 1)"),
        ("get_basis", "from statsforecast._lib import mfles; import numpy as np; mfles.get_basis(np.random.randn(500).reshape(-1,1), 3, -1.0, 0)"),
    ]

    print("\n--- Cold Start Comparison (subprocess, includes import + first call) ---")
    print(f"{'Function':<20} {'Numba (ms)':<15} {'C++ (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    cwd = str(Path(__file__).resolve().parent)
    python = sys.executable

    for (name, numba_code), (_, cpp_code) in zip(funcs, cpp_funcs):
        # Numba cold start
        numba_times = []
        for _ in range(3):
            cmd = f"import time; t0=time.perf_counter(); {numba_code}; print(f'{{(time.perf_counter()-t0)*1000:.2f}}')"
            r = subprocess.run(
                [python, "-c", cmd],
                capture_output=True, text=True, cwd=cwd,
                timeout=60,
            )
            if r.returncode == 0:
                numba_times.append(float(r.stdout.strip()))

        # C++ cold start
        cpp_times = []
        for _ in range(3):
            cmd = f"import time; t0=time.perf_counter(); {cpp_code}; print(f'{{(time.perf_counter()-t0)*1000:.2f}}')"
            r = subprocess.run(
                [python, "-c", cmd],
                capture_output=True, text=True, cwd=cwd,
                timeout=60,
            )
            if r.returncode == 0:
                cpp_times.append(float(r.stdout.strip()))

        if numba_times and cpp_times:
            n_ms = min(numba_times)
            c_ms = min(cpp_times)
            spd = n_ms / c_ms if c_ms > 0 else float("inf")
            print(f"{name:<20} {n_ms:<15.1f} {c_ms:<15.1f} {spd:<10.1f}x")
        else:
            print(f"{name:<20} {'ERROR':<15} {'ERROR':<15}")


# ===================================================================
# Sequence Length Scaling Benchmarks
# ===================================================================

SWEEP_LENGTHS = [50, 100, 250, 500, 1000, 2500, 5000, 10000]


def _make_seasonal_series(n, m=12, seed=42):
    """Generate a synthetic seasonal series of length n."""
    rng = np.random.RandomState(seed)
    trend = np.linspace(100, 100 + n * 0.05, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / m)
    noise = rng.randn(n) * 5
    return (trend + seasonal + noise).astype(np.float64)


def benchmark_scaling():
    """Benchmark key functions across a range of sequence lengths."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsforecast.tbats import makeTBATSWMatrix, makeTBATSGMatrix

    scaling_results = {}  # model -> [(n, old_ms, new_ms, speedup, match)]

    for n in SWEEP_LENGTHS:
        y = _make_seasonal_series(n)
        m = 12

        # --- SES SSE ---
        alpha = 0.3
        warmup_numba(old_ses_sse, alpha, y)
        old_ms, old_res = bench(old_ses_sse, alpha, y)
        new_ms, new_res = bench(_ses.ses_sse, alpha, y)
        ok, _ = check_close(old_res, new_res)
        scaling_results.setdefault("ses_sse", []).append(
            (n, old_ms, new_ms, old_ms / new_ms if new_ms > 0 else 0, ok))

        # --- GARCH sigma2 ---
        coeff = np.array([0.1, 0.2, 0.7])
        warmup_numba(old_garch_sigma2, coeff, y, 1, 1)
        old_ms, old_s2 = bench(old_garch_sigma2, coeff, y, 1, 1)
        new_ms, new_s2 = bench(_garch.compute_sigma2, coeff, y, 1, 1)
        ok, _ = check_close(old_s2, new_s2)
        scaling_results.setdefault("garch_sigma2", []).append(
            (n, old_ms, new_ms, old_ms / new_ms if new_ms > 0 else 0, ok))

        # --- GARCH loglik ---
        warmup_numba(old_garch_loglik, coeff, y, 1, 1)
        old_ms, old_ll = bench(old_garch_loglik, coeff, y, 1, 1)
        new_ms, new_ll = bench(_garch.loglik, coeff, y, 1, 1)
        ok, _ = check_close(old_ll, new_ll)
        scaling_results.setdefault("garch_loglik", []).append(
            (n, old_ms, new_ms, old_ms / new_ms if new_ms > 0 else 0, ok))

        # --- CES cescalc (PARTIAL) ---
        alpha_0, alpha_1, beta_0, beta_1 = 1.5, 1.0, 0.9, 0.0
        season = 2  # PARTIAL
        nmse = 3
        lags = m
        init_states = np.zeros((lags, 3), dtype=np.float32)
        init_states[:, 0] = np.mean(y[:lags]).astype(np.float32)
        init_states[:, 1] = (init_states[:, 0] / 1.1).astype(np.float32)
        try:
            init_states[:, 2] = seasonal_decompose(
                y, period=lags).seasonal[:lags].astype(np.float32)
        except Exception:
            init_states[:, 2] = 0.0

        def run_old_ces():
            s = np.zeros((m + n, 3), dtype=np.float32)
            s[:m] = init_states.copy()
            e = np.zeros(n)
            amse = np.zeros(30)
            old_cescalc(y.copy(), s, m, season, alpha_0, alpha_1, beta_0,
                        beta_1, e, amse, nmse, 1)
            return s, e

        def run_new_ces():
            s = np.zeros((m + n, 3), dtype=np.float32)
            s[:m] = init_states.copy()
            e = np.zeros(n)
            amse = np.zeros(30)
            _ces.cescalc(y.copy(), s, m, season, float(alpha_0),
                         float(alpha_1), float(beta_0), float(beta_1),
                         e, amse, nmse, 1)
            return s, e

        warmup_numba(old_cescalc, y.copy(),
                     np.zeros((m + n, 3), dtype=np.float32),
                     m, season, alpha_0, alpha_1, beta_0, beta_1,
                     np.zeros(n), np.zeros(30), nmse, 1)
        num = max(1, 20 // max(1, n // 500))
        old_ms, (old_s, old_e) = bench(run_old_ces, number=num, repeat=3)
        new_ms, (new_s, new_e) = bench(run_new_ces, number=num, repeat=3)
        ok1, _ = check_close(old_s, new_s, rtol=1e-3, atol=1e-2)
        ok2, _ = check_close(old_e, new_e, rtol=1e-3, atol=1e-2)
        scaling_results.setdefault("cescalc", []).append(
            (n, old_ms, new_ms, old_ms / new_ms if new_ms > 0 else 0,
             ok1 and ok2))

        # --- TBATS calcFaster ---
        seasonal_periods = np.array([12.0])
        k_vector = np.array([3])
        tau = int(2 * np.sum(k_vector))
        phi, tb_alpha, tb_beta = 0.99, 0.1, 0.05
        ar_coeffs = np.array([0.3, 0.1])
        ma_coeffs = np.array([0.2])
        gamma_bold = np.random.RandomState(42).randn(1, tau)

        F_mat = np.asarray(
            _tbats.makeFMatrix(phi, tau, tb_alpha, tb_beta, ar_coeffs,
                               ma_coeffs, gamma_bold, seasonal_periods,
                               k_vector),
            dtype=np.float64)
        p, q = len(ar_coeffs), len(ma_coeffs)
        w = makeTBATSWMatrix(phi, k_vector, ar_coeffs, ma_coeffs, tau)
        g, _ = makeTBATSGMatrix(k_vector, tb_alpha, 1, tb_beta,
                                np.array([0.01] * len(k_vector)),
                                np.array([0.01] * len(k_vector)),
                                p, q, tau)
        x0 = np.zeros(F_mat.shape[0])
        x0[0] = np.mean(y[:12])
        w_t = np.asarray(w, dtype=np.float64)
        g_mat = np.asarray(g, dtype=np.float64)

        warmup_numba(old_calcTBATSFaster, y, w_t, g_mat, F_mat, x0)
        old_ms, (old_yh, old_er, _) = bench(
            old_calcTBATSFaster, y, w_t, g_mat, F_mat, x0,
            number=num, repeat=3)
        new_ms, (new_yh, new_er, _) = bench(
            _tbats.calcFaster, y, w_t.ravel(), g_mat, F_mat, x0,
            number=num, repeat=3)
        ok, _ = check_close(old_yh, new_yh)
        scaling_results.setdefault("calcFaster", []).append(
            (n, old_ms, new_ms, old_ms / new_ms if new_ms > 0 else 0, ok))

        # --- MFLES get_basis ---
        warmup_numba(old_get_basis, y, 3)
        old_ms, old_b = bench(old_get_basis, y, 3, number=num, repeat=3)
        new_ms, new_b = bench(
            _mfles.get_basis, y.reshape(-1, 1), 3, -1.0, 0,
            number=num, repeat=3)
        ok, _ = check_close(old_b, new_b)
        scaling_results.setdefault("get_basis", []).append(
            (n, old_ms, new_ms, old_ms / new_ms if new_ms > 0 else 0, ok))

        # --- MFLES siegel_repeated_medians ---
        x_idx = np.arange(n, dtype=np.float64)
        warmup_numba(old_siegel_repeated_medians, x_idx, y)
        srm_num = max(1, num // 2)
        old_ms, old_med = bench(
            old_siegel_repeated_medians, x_idx, y,
            number=srm_num, repeat=3)
        new_ms, new_med = bench(
            _mfles.siegel_repeated_medians,
            x_idx.reshape(-1, 1), y.reshape(-1, 1),
            number=srm_num, repeat=3)
        ok, _ = check_close(old_med, new_med)
        scaling_results.setdefault("siegel_medians", []).append(
            (n, old_ms, new_ms, old_ms / new_ms if new_ms > 0 else 0, ok))

    return scaling_results


def print_scaling_results(scaling_results):
    """Print scaling benchmark results as tables."""
    print(f"\n{'=' * 110}")
    print("  SEQUENCE LENGTH SCALING BENCHMARKS")
    print(f"{'=' * 110}")

    # Header with lengths
    lengths = SWEEP_LENGTHS
    hdr = f"{'Model':<18}"
    for n in lengths:
        hdr += f"  {'n=' + str(n):>12}"
    print(f"\n{hdr}")
    print("-" * (18 + 14 * len(lengths)))

    # Speedup table
    print("\n  Speedup (old_numba / new_cpp):")
    print(f"{'Model':<18}", end="")
    for n in lengths:
        print(f"  {'n=' + str(n):>12}", end="")
    print()
    print("-" * (18 + 14 * len(lengths)))

    for model in scaling_results:
        row = scaling_results[model]
        print(f"{model:<18}", end="")
        for _, _, _, spd, ok in row:
            flag = "" if ok else "!"
            print(f"  {spd:>11.2f}x{flag}", end="")
        print()

    # Absolute time table (new C++)
    print(f"\n  Absolute time — new C++ (ms):")
    print(f"{'Model':<18}", end="")
    for n in lengths:
        print(f"  {'n=' + str(n):>12}", end="")
    print()
    print("-" * (18 + 14 * len(lengths)))

    for model in scaling_results:
        row = scaling_results[model]
        print(f"{model:<18}", end="")
        for _, _, new_ms, _, _ in row:
            print(f"  {new_ms:>12.3f}", end="")
        print()

    # Absolute time table (old numba)
    print(f"\n  Absolute time — old numba (ms):")
    print(f"{'Model':<18}", end="")
    for n in lengths:
        print(f"  {'n=' + str(n):>12}", end="")
    print()
    print("-" * (18 + 14 * len(lengths)))

    for model in scaling_results:
        row = scaling_results[model]
        print(f"{model:<18}", end="")
        for _, old_ms, _, _, _ in row:
            print(f"  {old_ms:>12.3f}", end="")
        print()

    # Equivalence check
    all_ok = all(ok for model in scaling_results
                 for _, _, _, _, ok in scaling_results[model])
    total = sum(len(v) for v in scaling_results.values())
    n_pass = sum(1 for model in scaling_results
                 for _, _, _, _, ok in scaling_results[model] if ok)
    print(f"\n  Output equivalence: {n_pass}/{total} PASS")
    if not all_ok:
        print("  FAILURES:")
        for model in scaling_results:
            for n, _, _, _, ok in scaling_results[model]:
                if not ok:
                    print(f"    - {model} n={n}")


# ===================================================================
# Main
# ===================================================================

def print_results():
    # Group results by category
    lowlevel = [r for r in results if r["model"] not in
                ("ses_fitted", "croston_fitted", "garch_fitted", "ces_fitted",
                 "tbats_fitted", "ses_pi", "ces_pi")]
    fitted = [r for r in results if r["model"] in
              ("ses_fitted", "croston_fitted", "garch_fitted", "ces_fitted",
               "tbats_fitted")]
    pi = [r for r in results if r["model"] in
          ("ses_pi", "ces_pi")]

    def _print_section(title, rows):
        if not rows:
            return
        print(f"\n{'=' * 100}")
        print(f"  {title}")
        print(f"{'=' * 100}")
        print(f"{'Model':<20} {'Series':<16} {'Old (ms)':<12} {'New (ms)':<12} "
              f"{'Speedup':<10} {'Match':<8} {'Notes'}")
        print("-" * 100)
        for r in rows:
            match_str = "PASS" if r["match"] else "FAIL"
            print(f"{r['model']:<20} {r['series']:<16} {r['old_ms']:<12.3f} "
                  f"{r['new_ms']:<12.3f} {r['speedup']:<10.2f}x {match_str:<8} "
                  f"{r['notes']}")
        print("-" * 100)

    _print_section("LOW-LEVEL FUNCTION BENCHMARKS", lowlevel)
    _print_section("FITTED VALUES BENCHMARKS", fitted)
    _print_section("PREDICTION INTERVALS BENCHMARKS", pi)

    # Summary
    n_pass = sum(1 for r in results if r["match"])
    n_total = len(results)
    speedups = [r["speedup"] for r in results]
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}")
    print(f"Output equivalence: {n_pass}/{n_total} PASS")
    print(f"Speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
    print(f"Median speedup: {np.median(speedups):.2f}x")
    if n_pass < n_total:
        print("\nFAILED benchmarks:")
        for r in results:
            if not r["match"]:
                print(f"  - {r['model']} ({r['series']}): {r['notes']}")


if __name__ == "__main__":
    print("Generating time series with utilsforecast...")
    print(f"Series length: {N}, Seed: {SEED}\n")

    benchmarks = [
        # --- Low-level function benchmarks ---
        ("SES SSE", benchmark_ses_sse),
        ("SES Forecast", benchmark_ses_forecast),
        ("Expand Fitted (Intermittent)", benchmark_expand_fitted),
        ("GARCH", benchmark_garch),
        ("CES", benchmark_ces),
        ("TBATS", benchmark_tbats),
        ("MFLES", benchmark_mfles),
        # --- Fitted values benchmarks ---
        ("SES Fitted (full pipeline)", benchmark_ses_fitted),
        ("Croston Fitted", benchmark_croston_fitted),
        ("GARCH Fitted (full pipeline)", benchmark_garch_fitted),
        ("CES Fitted", benchmark_ces_fitted),
        ("TBATS Fitted", benchmark_tbats_fitted),
        # --- Prediction intervals benchmarks ---
        ("SES Prediction Intervals", benchmark_ses_pi),
        ("CES Prediction Intervals (MC)", benchmark_ces_pi),
    ]

    for name, fn in benchmarks:
        print(f"Running {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print_results()

    print("\nRunning sequence length scaling benchmarks...")
    try:
        scaling = benchmark_scaling()
        print_scaling_results(scaling)
    except Exception as e:
        print(f"Scaling benchmark error: {e}")
        import traceback
        traceback.print_exc()

    print("\nRunning cold-start comparison...")
    try:
        benchmark_cold_start()
    except Exception as e:
        print(f"Cold-start benchmark error: {e}")
