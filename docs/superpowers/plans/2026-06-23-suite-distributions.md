# Suite-Wide Error Distributions (CES & Theta) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the non-normal error distributions already in ARIMA (PR #1113) and ETS (PR #1116) — `normal`, `laplace`, `t`, `skew-normal`, `ged` — to AutoCES and AutoTheta, sharing one C++ header and one Python module so future methods are a small recipe.

**Architecture:** A header-only C++ file `include/statsforecast/distributions.h` holds the `Distribution` enum and the per-observation negative-log-likelihood (NLL) functions, extracted verbatim from the working ETS code. A Python module `python/statsforecast/distributions.py` is the single import hub: it re-exports the (post-merge) enums/quantiles from `utils.py` and adds shared fitting helpers (init params, param extraction, AIC/BIC/AICc correction). Each method fits in ONE pass: Python assembles an extended parameter vector `[structural params..., distribution params...]`, and a C++ objective runs a single Nelder-Mead minimizing the shared NLL. ETS is refactored to use the shared layers *without behavior change* (proving the infra); CES and Theta then reuse the same pattern.

**Tech Stack:** C++20 (pybind11, Eigen, `nm::NelderMead`), Python 3.9+ (numpy, scipy.stats), pytest. Build via `uv run python setup.py build_ext --inplace`.

## Global Constraints

- **Run Python with `uv run python`** (never bare `python`). Build: `uv run python setup.py build_ext --inplace`. Tests: `uv run python -m pytest tests/`.
- **Branch:** `feat/ets_likelihood`. `main` (with ARIMA #1113) is ALREADY merged in (commit `98e922a`); `utils.py` now defines `class Distribution(str, Enum)`, `class ArimaMethod(str, Enum)`, `_VALID_DISTRIBUTIONS = tuple(Distribution)`, and an eager-dict `_quantiles`.
- **Five distributions only:** `normal`, `laplace`, `t`, `skew-normal`, `ged`. The string⇄enum names map: `normal→Normal`, `laplace→Laplace`, `t→StudentT`, `skew-normal→SkewNormal`, `ged→GED`.
- **One-pass fitting everywhere** (decision B): Python seeds the extended `x0`; C++ runs a single Nelder-Mead. No internal C++ warm-start pass. Do NOT refactor ETS's optimizer structure beyond swapping in the shared NLL.
- **`nm::NelderMead` MINIMIZES.** Distribution objectives return the per-obs NLL and MUST return `std::numeric_limits<double>::infinity()` for invalid/degenerate points (matching ETS's `ObjectiveFunctionDist`), never `-inf`.
- **Parameter layout (keep existing, decision F):** optimizer-vector tail per distribution — `laplace`: none (`b_hat` analytic); `t`: `[log_sigma2, log(nu-2)]`; `skew-normal`: `[log_sigma2, alpha]`; `ged`: `[log_sigma, log_beta]` (note GED stores HALF-log = `log σ`, so `sigma2 = exp(x)**2`). Do NOT migrate GED to `log_sigma2` (both ETS and ARIMA use `log_sigma`; migrating churns working code).
- **C++ header include:** place at `include/statsforecast/distributions.h`, include BARE as `#include "distributions.h"` (the include dir is `include/statsforecast`). No `setup.py` change. New `.cpp` files are auto-built by `glob("src/*.cpp")`.
- **Model-dict key contract:** every fitted model dict has `distribution`; add `nu` only for `t`, `alpha_dist` only for `skew-normal`, `beta_dist` only for `ged`; `normal`/`laplace` add neither.
- **Distribution is user-fixed** (decision E): `auto_ces`/`auto_theta` select only the structural type (seasontype / modeltype); they do NOT search over distributions.
- After each task: build the extension if C++ changed, run the named tests, and **commit**.

---

## Task 1: Shared C++ distribution header

**Files:**
- Create: `include/statsforecast/distributions.h`
- Test: exercised indirectly by Task 3 (ETS numerical-equivalence); a compile check is the deliverable here.

**Interfaces:**
- Produces (used by Tasks 3, 5, 6):
  - `enum class Distribution { Normal=0, Laplace=1, StudentT=2, SkewNormal=3, GED=4 };`
  - `int distribution_n_extra_params(Distribution d);`
  - `double negloglik_laplace(const double* e, int n);`
  - `double negloglik_t(const double* e, int n, double log_sigma2, double log_nu_m2);`
  - `double negloglik_skewnorm(const double* e, int n, double log_sigma2, double alpha);`
  - `double negloglik_ged(const double* e, int n, double log_sigma, double log_beta);`
  - Each `negloglik_*` returns the per-observation-averaged NLL CORE ONLY (no multiplicative-error Jacobian; the caller adds that if needed).

- [ ] **Step 1: Write the header**

Create `include/statsforecast/distributions.h`. The math is lifted verbatim from `src/ets.cpp:359-403` (the working ETS objective), refactored so the residual pointer and the distribution scalars are passed in by the caller:

```cpp
#pragma once

#include <cmath>
#include <limits>
#include <numbers>

namespace dist {

enum class Distribution {
  Normal = 0,
  Laplace = 1,
  StudentT = 2,
  SkewNormal = 3,
  GED = 4,
};

// Number of distribution params appended to the optimizer vector.
// Laplace: 0 (scale b_hat is analytic). All others: 2.
inline int distribution_n_extra_params(Distribution d) {
  return (d == Distribution::Laplace) ? 0 : 2;
}

// Per-observation negative log-likelihood CORES.
// e: residual array (additive errors), length n. Returns +inf on degeneracy.

inline double negloglik_laplace(const double *e, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += std::abs(e[i]);
  double b_hat = s / static_cast<double>(n);
  if (b_hat <= 0.0)
    return std::numeric_limits<double>::infinity();
  return std::log(b_hat);
}

inline double negloglik_t(const double *e, int n, double log_sigma2,
                          double log_nu_m2) {
  double sigma2 = std::exp(log_sigma2);
  double nu = std::exp(log_nu_m2) + 2.0;
  double half_nu1 = 0.5 * (nu + 1.0);
  double sum_log_kernel = 0.0;
  for (int i = 0; i < n; ++i)
    sum_log_kernel += std::log(e[i] * e[i] / (nu * sigma2) + 1.0);
  return (0.5 * log_sigma2 + std::lgamma(nu / 2.0) - std::lgamma(half_nu1) +
          0.5 * std::log(nu * std::numbers::pi) +
          half_nu1 / static_cast<double>(n) * sum_log_kernel);
}

inline double negloglik_skewnorm(const double *e, int n, double log_sigma2,
                                 double alpha) {
  double sigma = std::exp(0.5 * log_sigma2);
  double sum_sq = 0.0;
  double sum_log_cdf = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += e[i] * e[i];
    double z = alpha * e[i] / (sigma * std::numbers::sqrt2);
    sum_log_cdf += std::log(0.5 * std::erfc(-z) + 1e-30);
  }
  return (-std::log(2.0) + 0.5 * std::log(2.0 * std::numbers::pi) +
          0.5 * log_sigma2 +
          sum_sq / (2.0 * static_cast<double>(n) * sigma * sigma) -
          sum_log_cdf / static_cast<double>(n));
}

inline double negloglik_ged(const double *e, int n, double log_sigma,
                            double log_beta) {
  double sigma = std::exp(log_sigma);
  double beta_ged = std::exp(log_beta);
  double sum_pow = 0.0;
  for (int i = 0; i < n; ++i)
    sum_pow += std::pow(std::abs(e[i]) / sigma, beta_ged);
  return (std::log(2.0) + log_sigma + std::lgamma(1.0 / beta_ged) - log_beta +
          sum_pow / static_cast<double>(n));
}

} // namespace dist
```

- [ ] **Step 2: Add a temporary translation-unit compile check**

`distributions.h` is header-only with no current includer, so confirm it compiles under C++20. Temporarily add `#include "distributions.h"` near the top of `src/statsforecast.cpp` (the module entry point, which is already a compiled source).

- [ ] **Step 3: Build and verify it compiles**

Run: `uv run python setup.py build_ext --inplace`
Expected: build completes with no errors (warnings about unused functions are acceptable; the functions are `inline`).

- [ ] **Step 4: Commit**

```bash
git add include/statsforecast/distributions.h src/statsforecast.cpp
git commit -m "feat(cpp): add shared distributions.h header (enum + NLL cores)"
```

---

## Task 2: Shared Python distribution module

**Files:**
- Create: `python/statsforecast/distributions.py`
- Test: `tests/test_distributions.py` (created here; expanded in Task 8)

**Interfaces:**
- Consumes from `utils.py` (already present post-merge): `Distribution`, `ArimaMethod`, `_VALID_DISTRIBUTIONS`, `_quantiles`, `_calculate_intervals`.
- Produces (used by Tasks 3–8):
  - `VALID_DISTRIBUTIONS` (= `_VALID_DISTRIBUTIONS`)
  - `distribution_n_extra_params(distribution: str) -> int`
  - `switch_distribution(distribution: str, module) -> <module>.Distribution`
  - `dist_init_params(distribution: str, var_init: float) -> tuple[int, list[float]]` (returns `(n_dist, dist_init)`)
  - `extract_dist_params(distribution: str, fit_par_dist, residuals=None) -> dict`
  - `aic_bic_aicc(neg2logL: float, np_eff: int, n: int) -> tuple[float, float, float]`
  - Re-exports: `Distribution`, `_quantiles`, `_calculate_intervals`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_distributions.py`:

```python
import numpy as np
import pytest
from scipy import stats

from statsforecast import distributions as D


def test_valid_distributions_matches_utils():
    from statsforecast.utils import _VALID_DISTRIBUTIONS
    assert D.VALID_DISTRIBUTIONS == _VALID_DISTRIBUTIONS
    assert set(str(d) for d in D.VALID_DISTRIBUTIONS) == {
        "normal", "laplace", "t", "skew-normal", "ged"
    }


def test_n_extra_params():
    assert D.distribution_n_extra_params("laplace") == 0
    assert D.distribution_n_extra_params("normal") == 0
    for d in ("t", "skew-normal", "ged"):
        assert D.distribution_n_extra_params(d) == 2


def test_dist_init_params_layout():
    var = 4.0
    assert D.dist_init_params("laplace", var) == (0, [])
    n_t, init_t = D.dist_init_params("t", var)
    assert n_t == 2
    assert init_t == [np.log(var), np.log(3.0)]
    n_sn, init_sn = D.dist_init_params("skew-normal", var)
    assert init_sn == [np.log(var), 0.0]
    n_ged, init_ged = D.dist_init_params("ged", var)
    assert init_ged == [0.5 * np.log(var), np.log(2.0)]


def test_extract_dist_params():
    # t: tail = [log_sigma2, log(nu-2)]
    out = D.extract_dist_params("t", np.array([np.log(9.0), np.log(3.0)]))
    assert out["nu"] == pytest.approx(5.0)
    assert out["sigma2"] == pytest.approx(9.0)
    # skew-normal: tail = [log_sigma2, alpha]
    out = D.extract_dist_params("skew-normal", np.array([np.log(9.0), 1.5]))
    assert out["alpha_dist"] == pytest.approx(1.5)
    assert out["sigma2"] == pytest.approx(9.0)
    # ged: tail = [log_sigma, log_beta], sigma2 = exp(x)**2
    out = D.extract_dist_params("ged", np.array([np.log(3.0), np.log(2.0)]))
    assert out["beta_dist"] == pytest.approx(2.0)
    assert out["sigma2"] == pytest.approx(9.0)
    # laplace: sigma2 from residuals, no shape key
    out = D.extract_dist_params("laplace", np.array([]),
                                residuals=np.array([-1.0, 1.0, -1.0, 1.0]))
    assert out["sigma2"] == pytest.approx(2.0)  # 2 * b_hat**2, b_hat = 1


def test_quantiles_match_scipy():
    level = np.array([80, 95])
    p = 0.5 + level / 200
    np.testing.assert_allclose(D._quantiles(level, "t", {"nu": 5.0}),
                               stats.t.ppf(p, df=5.0))
    np.testing.assert_allclose(D._quantiles(level, "laplace"),
                               stats.laplace.ppf(p))
    np.testing.assert_allclose(D._quantiles(level, "skew-normal", {"alpha_dist": 1.5}),
                               stats.skewnorm.ppf(p, a=1.5))
    np.testing.assert_allclose(D._quantiles(level, "ged", {"beta_dist": 1.2}),
                               stats.gennorm.ppf(p, beta=1.2))


def test_switch_distribution_maps_all():
    from statsforecast._lib import ets as _ets
    assert D.switch_distribution("t", _ets) == _ets.Distribution.StudentT
    assert D.switch_distribution("ged", _ets) == _ets.Distribution.GED
    with pytest.raises(ValueError):
        D.switch_distribution("cauchy", _ets)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m pytest tests/test_distributions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'statsforecast.distributions'` (and `_ets.Distribution` test will also fail until Task 3, so expect that one to error too).

- [ ] **Step 3: Write `distributions.py`**

Create `python/statsforecast/distributions.py`. Helper bodies are lifted from the working ETS code (`ets.py:714-723` for init, `818-823/864-869` for extraction, `794-832` for AIC):

```python
"""Single hub for error-distribution definitions and MLE-fitting helpers.

Re-exports the canonical Distribution enum and quantile utilities from
`utils.py` and adds the helpers shared by every distribution-aware model
(ARIMA, ETS, CES, Theta).
"""

import numpy as np

from .utils import (
    ArimaMethod,
    Distribution,
    _VALID_DISTRIBUTIONS,
    _calculate_intervals,
    _quantiles,
)

__all__ = [
    "Distribution",
    "ArimaMethod",
    "VALID_DISTRIBUTIONS",
    "distribution_n_extra_params",
    "switch_distribution",
    "dist_init_params",
    "extract_dist_params",
    "aic_bic_aicc",
    "_quantiles",
    "_calculate_intervals",
]

VALID_DISTRIBUTIONS = _VALID_DISTRIBUTIONS


def distribution_n_extra_params(distribution) -> int:
    """Number of distribution params appended to the optimizer vector."""
    return 0 if distribution in ("laplace", "normal") else 2


def switch_distribution(distribution: str, module):
    """Map a distribution string to a C++ module's Distribution enum.

    `module` is one of statsforecast._lib.{ets, ces, theta}.
    """
    mapping = {
        "normal": "Normal",
        "laplace": "Laplace",
        "t": "StudentT",
        "skew-normal": "SkewNormal",
        "ged": "GED",
    }
    if distribution not in mapping:
        raise ValueError(f"Unknown distribution: {distribution!r}")
    return getattr(module.Distribution, mapping[distribution])


def dist_init_params(distribution: str, var_init: float):
    """Return (n_dist, dist_init) for the optimizer-vector tail.

    Layout (must match include/statsforecast/distributions.h):
      laplace     -> (0, [])
      t           -> (2, [log(var), log(3.0)])           # nu_init = 5
      skew-normal -> (2, [log(var), 0.0])
      ged         -> (2, [0.5*log(var), log(2.0)])       # GED stores log(sigma)
    """
    if distribution == "t":
        return 2, [np.log(var_init), np.log(3.0)]
    if distribution == "skew-normal":
        return 2, [np.log(var_init), 0.0]
    if distribution == "ged":
        return 2, [0.5 * np.log(var_init), np.log(2.0)]
    return 0, []  # laplace / normal


def extract_dist_params(distribution: str, fit_par_dist, residuals=None) -> dict:
    """Convert the fitted optimizer tail into model-dict keys.

    Returns a dict with `sigma2` plus the shape key for the distribution:
      t           -> {"nu", "sigma2"}
      skew-normal -> {"alpha_dist", "sigma2"}
      ged         -> {"beta_dist", "sigma2"}     # sigma2 = exp(log_sigma)**2
      laplace     -> {"sigma2"}                  # from residuals; b_hat = mean(|e|)
      normal      -> {}
    """
    if distribution == "t":
        return {
            "nu": float(np.exp(fit_par_dist[1]) + 2.0),
            "sigma2": float(np.exp(fit_par_dist[0])),
        }
    if distribution == "skew-normal":
        return {
            "alpha_dist": float(fit_par_dist[1]),
            "sigma2": float(np.exp(fit_par_dist[0])),
        }
    if distribution == "ged":
        return {
            "beta_dist": float(np.exp(fit_par_dist[1])),
            "sigma2": float(np.exp(fit_par_dist[0])) ** 2,
        }
    if distribution == "laplace":
        b_hat = float(np.nanmean(np.abs(residuals)))
        return {"sigma2": 2.0 * b_hat ** 2}
    return {}


def aic_bic_aicc(neg2logL: float, np_eff: int, n: int):
    """Standard information criteria from -2*logLik and effective param count."""
    aic = neg2logL + 2 * np_eff
    bic = neg2logL + np.log(n) * np_eff
    if n - np_eff - 1 != 0.0:
        aicc = aic + 2 * np_eff * (np_eff + 1) / (n - np_eff - 1)
    else:
        aicc = np.inf
    return aic, bic, aicc
```

- [ ] **Step 4: Run the test to verify it passes (except the `_ets` enum case)**

Run: `uv run python -m pytest tests/test_distributions.py -q -k "not switch_distribution"`
Expected: PASS. (`test_switch_distribution_maps_all` needs the C++ enum binding from Task 3; it stays failing until then.)

- [ ] **Step 5: Commit**

```bash
git add python/statsforecast/distributions.py tests/test_distributions.py
git commit -m "feat: add shared distributions.py hub + fitting helpers"
```

---

## Task 3: Refactor ETS to the shared layers (no behavior change — proves the infra)

**Files:**
- Modify: `src/ets.cpp` (remove the local `Distribution` enum + inline NLL math; include the shared header)
- Modify: `python/statsforecast/ets.py` (import `switch_distribution`/helpers from `distributions.py`; replace inline init/extract/AIC)
- Test: `tests/test_ets.py` (existing distribution tests at lines 186-265 are the regression guard)

**Interfaces:**
- Consumes: `dist::Distribution`, `dist::negloglik_*` (Task 1); `switch_distribution`, `dist_init_params`, `extract_dist_params`, `aic_bic_aicc` (Task 2).
- Produces: the `_ets.Distribution` pybind enum used by `switch_distribution(..., _ets)` and Task 2's test.

- [ ] **Step 1: Confirm the regression guard passes BEFORE refactoring**

Run: `uv run python -m pytest tests/test_ets.py -q -k distribution`
Expected: the 6 ETS distribution tests PASS (this is the baseline the refactor must preserve).

- [ ] **Step 2: Refactor `src/ets.cpp` to use the shared header**

At the top, add `#include "distributions.h"`. Delete the local `enum class Distribution {...}` (ets.cpp:22-28) and replace it with `using dist::Distribution;`. In `ObjectiveFunctionDist`, keep the state-reconstruction scaffolding, the `Calc` call, and the `sumlog_adj` multiplicative-error term, but replace the inline per-distribution math (the `switch` at ets.cpp:359-406) with calls to the shared cores, adding `sumlog_adj`:

```cpp
  switch (distribution) {
  case Distribution::Laplace:
    return dist::negloglik_laplace(e.data(), static_cast<int>(n)) + sumlog_adj;
  case Distribution::StudentT:
    return dist::negloglik_t(e.data(), static_cast<int>(n),
                             params(n_total - 2), params(n_total - 1)) +
           sumlog_adj;
  case Distribution::SkewNormal:
    return dist::negloglik_skewnorm(e.data(), static_cast<int>(n),
                                    params(n_total - 2), params(n_total - 1)) +
           sumlog_adj;
  case Distribution::GED:
    return dist::negloglik_ged(e.data(), static_cast<int>(n),
                               params(n_total - 2), params(n_total - 1)) +
           sumlog_adj;
  default:
    return lik;
  }
```

Update the pybind enum binding (ets.cpp:453-458) to bind `dist::Distribution` (the values and names are unchanged: `Normal/Laplace/StudentT/SkewNormal/GED`). Keep `OptimizeDist` exactly as-is (one-pass).

- [ ] **Step 3: Build and run the ETS regression guard**

Run: `uv run python setup.py build_ext --inplace && uv run python -m pytest tests/test_ets.py -q -k distribution`
Expected: the same 6 tests PASS (numerical equivalence — the NLL cores are byte-for-byte the same math, so fits are identical).

- [ ] **Step 4: Refactor `python/statsforecast/ets.py` to use shared helpers**

Delete the local `switch_distribution` (ets.py:370-381) and import it: `from .distributions import switch_distribution, dist_init_params, extract_dist_params, aic_bic_aicc`. In `etsmodel`, replace the inline `dist_init` construction (ets.py:714-723) with `n_dist, dist_init = dist_init_params(distribution, var_init)`. Replace the inline shape-param extraction (ets.py:818-823, 864-869) with `extract_dist_params(distribution, fit_par_dist, residuals=e)` and read `nu`/`alpha_dist`/`beta_dist`/`sigma2` from the returned dict. Replace the inline AIC/BIC/AICc arithmetic (the three `aic=...; bic=...; aicc=...` blocks) with `aic, bic, aicc = aic_bic_aicc(neg2logL, np_eff, ny)`, keeping each distribution's existing `neg2logL` and `np_eff` computation (normal: `lik`, `np_+1`; laplace: `2*ny*fred.fn + ny*(2+log4)`, `np_+1`; t/skew/ged: `2*ny*fred.fn`, `np_+2`).

- [ ] **Step 5: Run the full ETS suite + the Task 2 enum test**

Run: `uv run python -m pytest tests/test_ets.py tests/test_distributions.py -q`
Expected: PASS (including `test_switch_distribution_maps_all`, now that `_ets.Distribution` is bound via the shared enum).

- [ ] **Step 6: Commit**

```bash
git add src/ets.cpp python/statsforecast/ets.py
git commit -m "refactor(ets): use shared distributions.h + distributions.py (no behavior change)"
```

---

## Task 4: Refactor ARIMA to the shared Python helpers (seam)

**Files:**
- Modify: `python/statsforecast/arima.py` (import shared helpers; align extraction/AIC; ARIMA fits in Python/scipy, so it uses only the Python module, not the C++ header)
- Test: `tests/test_arima.py`

**Interfaces:**
- Consumes: `Distribution`, `ArimaMethod`, `extract_dist_params`, `aic_bic_aicc`, `distribution_n_extra_params` (Task 2). ARIMA already imports `ArimaMethod, Distribution, _VALID_DISTRIBUTIONS, _quantiles` from `utils`.

- [ ] **Step 1: Confirm ARIMA baseline passes**

Run: `uv run python -m pytest tests/test_arima.py -q`
Expected: PASS (post-merge baseline).

- [ ] **Step 2: Point ARIMA's distribution imports at `distributions.py`**

In `arima.py`, change `from .utils import ArimaMethod, Distribution, _VALID_DISTRIBUTIONS, _quantiles` to import the same names from `.distributions` (which re-exports them). This makes `distributions.py` the single hub without moving the enum's physical home. Do NOT change ARIMA's scipy objective functions or its GED layout (it already uses `log_sigma` for GED, matching the shared convention).

- [ ] **Step 3: Replace ARIMA's inline param-extraction / AIC bookkeeping with the shared helpers**

Where `arima.py` builds the `coef` dict's distribution keys, call `extract_dist_params(distribution, fit_par_dist, residuals=...)` instead of the inline equivalent (it produces the identical `nu`/`alpha_dist`/`beta_dist`/`sigma2`). Optionally route ARIMA's IC arithmetic through `aic_bic_aicc(neg2logL, np_eff, n)` if it matches ARIMA's current formula. **Do NOT replace ARIMA's parameter-count logic with `distribution_n_extra_params`** — that helper returns the *optimizer-tail* size (0 for laplace, since `b_hat` is analytic), whereas ARIMA's AIC count is +1 for laplace (an estimated scale param) and +2 for t/skew/ged. Keep ARIMA's existing count exactly as #1113 computes it. The goal of this task is the import seam + shared `extract_dist_params`/`aic_bic_aicc`, not a numeric change.

- [ ] **Step 4: Run the ARIMA suite**

Run: `uv run python -m pytest tests/test_arima.py -q`
Expected: PASS (no numeric change — the helpers reproduce ARIMA's existing values).

- [ ] **Step 5: Commit**

```bash
git add python/statsforecast/arima.py
git commit -m "refactor(arima): import distribution helpers from distributions.py hub"
```

---

## Task 5: CES distribution fitting (one-pass, mirrors ETS)

**Files:**
- Modify: `src/ces.cpp` (add `Distribution` enum binding, `ces_target_fn_dist`, `optimize_dist`)
- Modify: `python/statsforecast/ces.py` (`distribution` param in `cesmodel`/`auto_ces`/`forecast_ces`/`ces_f`; rewire `_simulate_pred_intervals`)
- Modify: `python/statsforecast/models.py` (`AutoCES.__init__`/`fit`/`predict` thread `distribution`)
- Test: `tests/test_distributions.py`, `tests/test_ces.py`

**Interfaces:**
- Consumes: `dist::Distribution`, `dist::negloglik_*` (Task 1); `switch_distribution`, `dist_init_params`, `extract_dist_params`, `aic_bic_aicc`, `distribution_n_extra_params` (Task 2).
- Produces: `_ces.Distribution` enum, `_ces.optimize_dist`; `cesmodel(..., distribution=...)`; `AutoCES(distribution=...)`.
- CES is ADDITIVE-error: the NLL has no multiplicative-error Jacobian (`sumlog_adj = 0`). `cescalc` fills residual `e` in place; its `n*log(SSE)` return is ignored on the non-normal path.

- [ ] **Step 1: Add the CES distribution objective + optimizer in `src/ces.cpp`**

Add `#include "distributions.h"` at the top. After `ces_target_fn` (ces.cpp:276-307), add a distribution variant that reuses `cescalc` for in-place residuals and dispatches to the shared NLL (returns `+inf` on degeneracy — `nm::NelderMead` minimizes):

```cpp
double ces_target_fn_dist(const VectorXd &optimal_param, double init_alpha_0,
                          double init_alpha_1, double init_beta_0,
                          double init_beta_1, bool opt_alpha_0, bool opt_alpha_1,
                          bool opt_beta_0, bool opt_beta_1, const VectorXd &y,
                          int m, const RowMajorMatrixXd &init_states,
                          int n_components, int season, int nmse,
                          dist::Distribution distribution) {
  Eigen::Index n = y.size();
  RowMajorMatrixXd states = RowMajorMatrixXd::Zero(n + 2 * m, n_components);
  states.topRows(m) = init_states;

  Eigen::Index j = 0;
  double alpha_0 = opt_alpha_0 ? optimal_param[j++] : init_alpha_0;
  double alpha_1 = opt_alpha_1 ? optimal_param[j++] : init_alpha_1;
  double beta_0 = opt_beta_0 ? optimal_param[j++] : init_beta_0;
  double beta_1 = opt_beta_1 ? optimal_param[j++] : init_beta_1;

  VectorXd e = VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  VectorXd amse =
      VectorXd::Constant(nmse, std::numeric_limits<double>::quiet_NaN());

  double lik = cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1,
                       e, amse, nmse, 1);
  if (std::isnan(lik) || std::fabs(lik + 99999) < 1e-7)
    return std::numeric_limits<double>::infinity();

  int nn = static_cast<int>(n);
  int n_total = static_cast<int>(optimal_param.size());
  switch (distribution) {
  case dist::Distribution::Laplace:
    return dist::negloglik_laplace(e.data(), nn);
  case dist::Distribution::StudentT:
    return dist::negloglik_t(e.data(), nn, optimal_param(n_total - 2),
                             optimal_param(n_total - 1));
  case dist::Distribution::SkewNormal:
    return dist::negloglik_skewnorm(e.data(), nn, optimal_param(n_total - 2),
                                    optimal_param(n_total - 1));
  case dist::Distribution::GED:
    return dist::negloglik_ged(e.data(), nn, optimal_param(n_total - 2),
                               optimal_param(n_total - 1));
  default:
    return lik;
  }
}
```

Add `optimize_dist` mirroring `optimize` (ces.cpp:309-327) but calling `ces_target_fn_dist` with a trailing `distribution`:

```cpp
nm::OptimResult
optimize_dist(const Eigen::Ref<const VectorXd> &x0,
              const Eigen::Ref<const VectorXd> &lower,
              const Eigen::Ref<const VectorXd> &upper, double init_alpha_0,
              double init_alpha_1, double init_beta_0, double init_beta_1,
              bool opt_alpha_0, bool opt_alpha_1, bool opt_beta_0,
              bool opt_beta_1, const Eigen::Ref<const VectorXd> &y, int m,
              const Eigen::Ref<const RowMajorMatrixXd> &init_states,
              int n_components, const std::string &seasontype, int nmse,
              dist::Distribution distribution) {
  int season = switch_ces(seasontype);
  RowMajorMatrixXd init_states_copy = init_states;
  return nm::NelderMead(ces_target_fn_dist, x0, lower, upper, 0.05, 1e-4, 1.0,
                        2.0, 0.5, 0.5, 1000, 1e-4, true, init_alpha_0,
                        init_alpha_1, init_beta_0, init_beta_1, opt_alpha_0,
                        opt_alpha_1, opt_beta_0, opt_beta_1, y, m,
                        init_states_copy, n_components, season, nmse,
                        distribution);
}
```

In `init()` (ces.cpp:380-391) add the enum + binding:

```cpp
  py::enum_<dist::Distribution>(ces_mod, "Distribution")
      .value("Normal", dist::Distribution::Normal)
      .value("Laplace", dist::Distribution::Laplace)
      .value("StudentT", dist::Distribution::StudentT)
      .value("SkewNormal", dist::Distribution::SkewNormal)
      .value("GED", dist::Distribution::GED);
  ces_mod.def("optimize_dist", &optimize_dist,
              py::call_guard<py::gil_scoped_release>());
```

- [ ] **Step 2: Build and smoke-test the binding**

Run: `uv run python setup.py build_ext --inplace && uv run python -c "from statsforecast._lib import ces; print(ces.Distribution.StudentT, hasattr(ces, 'optimize_dist'))"`
Expected: prints `Distribution.StudentT True`.

- [ ] **Step 3: Write the failing CES fitting test**

Append to `tests/test_distributions.py`:

```python
from statsforecast.utils import AirPassengers as ap


@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_ces_distribution_keys(distribution):
    from statsforecast.ces import cesmodel
    m = cesmodel(ap, m=12, seasontype="S", alpha_0=None, alpha_1=None,
                 beta_0=None, beta_1=None, nmse=3, distribution=distribution)
    assert m["distribution"] == distribution
    assert np.isfinite(m["loglik"]) and np.isfinite(m["aic"])
    assert ("nu" in m) == (distribution == "t")
    assert ("alpha_dist" in m) == (distribution == "skew-normal")
    assert ("beta_dist" in m) == (distribution == "ged")
```

(`cesmodel` needs `alpha_0=...` defaults of `None`; pass them through as the wrapper does. If `cesmodel`'s positional signature differs, call with keywords as shown.)

- [ ] **Step 4: Run to verify it fails**

Run: `uv run python -m pytest tests/test_distributions.py -q -k ces`
Expected: FAIL with `TypeError: cesmodel() got an unexpected keyword argument 'distribution'`.

- [ ] **Step 5: Thread `distribution` through `ces.py`**

Add `from .distributions import (switch_distribution, dist_init_params, extract_dist_params, aic_bic_aicc, distribution_n_extra_params)` and `from .simulation import sample_errors` (lazy import inside functions is the repo idiom; keep it inside the functions that use it).

In `cesmodel` (ces.py:187), add `distribution="normal"`. The optimization becomes an `if/else` that REPLACES (not augments) the existing `optimize_ces_target_fn` call — the non-normal branch is one-pass and seeds from the SAME initial `par` values the normal optimizer uses (no normal warm-start, per decision B). Structure:

```python
    if distribution == "normal":
        fred = optimize_ces_target_fn(...)   # the existing call, unchanged
        # ... existing par-update block from fred.x (lines 225-239) ...
    else:
        fred = _ces_optimize_dist_branch(...)   # the block below
```

The non-normal branch (note `par` still holds the INITIAL values from `initparamces` at this point — there is no prior normal fit):

```python
        # one-pass joint fit: [free smoothing params..., dist params...]
        smooth_x0 = np.array(
            [par[k] for k, v in optimize_params.items() if v], dtype=np.float64
        )
        var_init = max(float(np.nanvar(y)), 1e-10)
        n_dist, dist_init = dist_init_params(distribution, var_init)
        x0_ext = np.concatenate([smooth_x0, dist_init])
        all_lower = np.array([0.01, 0.01, 0.01, 0.01])
        all_upper = np.array([1.8, 1.9, 1.5, 1.5])
        n_smooth = len(smooth_x0)
        lower_ext = np.concatenate([all_lower[:n_smooth], np.full(n_dist, -np.inf)])
        upper_ext = np.concatenate([all_upper[:n_smooth], np.full(n_dist, np.inf)])
        opt_res = _ces.optimize_dist(
            x0_ext, lower_ext, upper_ext,
            float(par["alpha_0"]), float(par["alpha_1"]),
            float(par["beta_0"]), float(par["beta_1"]),
            bool(optimize_params["alpha_0"]), bool(optimize_params["alpha_1"]),
            bool(optimize_params["beta_0"]), bool(optimize_params["beta_1"]),
            np.asarray(y, dtype=np.float64), m,
            np.asarray(init_state, dtype=np.float64),
            n_components, seasontype, nmse,
            switch_distribution(distribution, _ces),
        )
        fred = results(*opt_res)
        fit_par = fred.x[:n_smooth] if n_dist > 0 else fred.x
        fit_par_dist = fred.x[-n_dist:] if n_dist > 0 else np.array([])
        # write the optimized smoothing params back into par (same order as normal path)
        jj = 0
        for key in ("alpha_0", "alpha_1", "beta_0", "beta_1"):
            if optimize_params[key]:
                par[key] = fit_par[jj]; jj += 1
```

Then compute residuals via the existing `pegelsresid_ces(... **par)` call (already at ces.py:241) so `e`, `states`, `amse` reflect the final params, and replace the IC block for the non-normal case:

```python
    ny = len(y)
    if distribution == "normal":
        np_ = n_components + 1
        aic = lik + 2 * np_
        bic = lik + np.log(ny) * np_
        aicc = aic + 2 * np_ * (np_ + 1) / (ny - np_ - 1) if ny - np_ - 1 != 0.0 else np.inf
        loglik = -0.5 * lik
        sigma2 = np.sum(e ** 2) / (ny - np_ - 1)
        dist_extra = {}
    else:
        n_dist_count = 1 if distribution == "laplace" else 2
        np_ = n_components + n_dist_count
        if distribution == "laplace":
            neg2logL = 2 * ny * fred.fn + ny * (2 + np.log(4))
        else:
            neg2logL = 2 * ny * fred.fn
        aic, bic, aicc = aic_bic_aicc(neg2logL, np_, ny)
        loglik = -0.5 * neg2logL
        dist_extra = extract_dist_params(distribution, fit_par_dist, residuals=e)
        sigma2 = dist_extra.pop("sigma2")
```

Add `distribution=distribution, **dist_extra` to the returned dict (and keep `sigma2=sigma2`). NOTE (documented limitation): the normal CES `lik` is the concentrated `n*log(SSE)`, NOT a full `-2logL`; non-normal `aic` is a full `-2logL`. They are therefore NOT comparable across distributions — which is fine because the distribution is user-fixed (decision E) and `auto_ces` only compares seasontypes within one distribution.

Thread `distribution="normal"` through `auto_ces` (ces.py:343 — add the param, pass to each `cesmodel(...)` call at line 387; selection by `ic` is unchanged), `forecast_ces`, and `ces_f`/the `AutoCES` call path.

- [ ] **Step 6: Rewire `_simulate_pred_intervals` to the fitted distribution**

Replace the hardcoded Gaussian draw (ces.py:309) so intervals reflect the fitted error law. Add a mapping helper to `distributions.py` (reused by Theta in Task 6):

```python
# in distributions.py
def error_params_from_model(model: dict) -> dict | None:
    """Map a fitted model dict's distribution params to sample_errors() params."""
    dist = model.get("distribution", "normal")
    if dist == "t":
        return {"df": model["nu"]}
    if dist == "skew-normal":
        return {"skewness": model["alpha_dist"]}
    if dist == "ged":
        return {"shape": model["beta_dist"]}
    return None  # normal / laplace -> sample_errors derives scale from sigma
```

Then in `_simulate_pred_intervals`:

```python
def _simulate_pred_intervals(model, h, level):
    from statsforecast.simulation import sample_errors
    from statsforecast.distributions import error_params_from_model
    rng = np.random.default_rng(1)
    nsim = 5000
    dist = model.get("distribution", "normal")
    sigma = np.sqrt(model["sigma2"])
    params = error_params_from_model(model)
    residuals = model.get("residuals", None)
    y_path = np.zeros([nsim, h])
    season = switch_ces(model["seasontype"])
    for k in range(nsim):
        e = sample_errors(size=model["states"].shape, sigma=sigma,
                          distribution=dist, params=params, residuals=residuals, rng=rng)
        states = model["states"]
        fcsts = np.zeros(h, dtype=np.float64)
        cesforecast(states=states + e, n=model["n"], m=model["m"],
                    season=season, h=h, f=fcsts, **model["par"])
        y_path[k,] = fcsts
    lower = np.quantile(y_path, 0.5 - np.array(level) / 200, axis=0)
    upper = np.quantile(y_path, 0.5 + np.array(level) / 200, axis=0)
    return {**{f"lo-{lv}": lower[i] for i, lv in enumerate(level)},
            **{f"hi-{lv}": upper[i] for i, lv in enumerate(level)}}
```

- [ ] **Step 7: Thread `distribution` through `AutoCES` in `models.py`**

Add `distribution: str = "normal"` to `AutoCES.__init__` (models.py:~994), store `self.distribution`, validate against `VALID_DISTRIBUTIONS`, and pass it to `auto_ces(...)` in `fit` (models.py:1030). `predict` needs no change (reads `model_["distribution"]` via `forecast_ces`).

- [ ] **Step 8: Run CES tests + interval ordering**

Append an interval-ordering test to `tests/test_distributions.py`:

```python
@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_ces_interval_ordering(distribution):
    from statsforecast.models import AutoCES
    model = AutoCES(season_length=12, model="S", distribution=distribution)
    model.fit(ap)
    assert model.model_["distribution"] == distribution
    pred = model.predict(h=12, level=[80, 95])
    assert np.all(pred["lo-95"] < pred["lo-80"])
    assert np.all(pred["lo-80"] < pred["mean"])
    assert np.all(pred["mean"] < pred["hi-80"])
    assert np.all(pred["hi-80"] < pred["hi-95"])
```

Run: `uv run python -m pytest tests/test_distributions.py tests/test_ces.py -q -k "ces or CES"`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/ces.cpp python/statsforecast/ces.py python/statsforecast/distributions.py python/statsforecast/models.py tests/test_distributions.py
git commit -m "feat(ces): distribution-aware fitting + distribution-aware prediction intervals"
```

---

## Task 6: Theta distribution fitting (C++ likelihood objective; non-normal only)

**Files:**
- Modify: `src/theta.cpp` (add `Distribution` enum, `target_fn_dist`, `optimize_dist`)
- Modify: `python/statsforecast/theta.py` (`distribution` in `thetamodel`/`auto_theta`/`forecast_theta`)
- Modify: `python/statsforecast/models.py` (`AutoTheta.__init__`/`fit`/`predict`)
- Test: `tests/test_distributions.py`, `tests/test_theta.py`

**Interfaces:**
- Consumes: Task 1 NLL cores, Task 2 helpers, `error_params_from_model` (added in Task 5).
- Produces: `_theta.Distribution`, `_theta.optimize_dist`; `thetamodel(..., distribution=...)`; `AutoTheta(distribution=...)`.
- **Theta likelihood uses `n_eff = n - 3` and residual slice `e[3:]`** (matching `calc`'s `e.tail(e.size()-3)`), and does NOT carry the `/mean_y` scale. Normal path is UNCHANGED (MSE fit + `mse` selection); only non-normal uses the new likelihood objective and AIC selection (decision D).

- [ ] **Step 1: Add the Theta distribution objective + optimizer in `src/theta.cpp`**

Add `#include "distributions.h"`. After `target_fn` (theta.cpp:123-153), add the distribution variant. It runs `calc` to fill `e`, then computes the NLL over `e[3:]` (no `/mean_y`), returning `+inf` on degeneracy:

```cpp
double target_fn_dist(const VectorXd &params, double init_level,
                      double init_alpha, double init_theta, bool opt_level,
                      bool opt_alpha, bool opt_theta, const VectorXd &y,
                      ModelType model_type, size_t nmse,
                      dist::Distribution distribution) {
  RowMajorMatrixXd states = RowMajorMatrixXd::Zero(y.size(), 5);
  size_t j = 0;
  double level = opt_level ? params[j++] : init_level;
  double alpha = opt_alpha ? params[j++] : init_alpha;
  double theta = opt_theta ? params[j++] : init_theta;
  VectorXd e = VectorXd::Zero(y.size());
  VectorXd amse = VectorXd::Zero(nmse);
  double mse = calc(y, states, model_type, level, alpha, theta, e, amse, nmse);
  if (std::isnan(mse) || std::abs(mse + 99999) < 1e-7)
    return std::numeric_limits<double>::infinity();

  int n_eff = static_cast<int>(y.size()) - 3;
  const double *e3 = e.data() + 3;
  int n_total = static_cast<int>(params.size());
  switch (distribution) {
  case dist::Distribution::Laplace:
    return dist::negloglik_laplace(e3, n_eff);
  case dist::Distribution::StudentT:
    return dist::negloglik_t(e3, n_eff, params(n_total - 2), params(n_total - 1));
  case dist::Distribution::SkewNormal:
    return dist::negloglik_skewnorm(e3, n_eff, params(n_total - 2),
                                    params(n_total - 1));
  case dist::Distribution::GED:
    return dist::negloglik_ged(e3, n_eff, params(n_total - 2),
                               params(n_total - 1));
  default:
    return mse;
  }
}
```

Add `optimize_dist` mirroring `optimize` (theta.cpp:155-175) but calling `target_fn_dist` with a trailing `distribution`:

```cpp
nm::OptimResult optimize_dist(const Eigen::Ref<const VectorXd> &x0,
                              const Eigen::Ref<const VectorXd> &lower,
                              const Eigen::Ref<const VectorXd> &upper,
                              double init_level, double init_alpha,
                              double init_theta, bool opt_level, bool opt_alpha,
                              bool opt_theta, const Eigen::Ref<const VectorXd> &y,
                              ModelType model_type, size_t nmse,
                              dist::Distribution distribution) {
  return nm::NelderMead(target_fn_dist, x0, lower, upper, 0.05, 1e-4, 1.0, 2.0,
                        0.5, 0.5, 1000, 1e-4, true, init_level, init_alpha,
                        init_theta, opt_level, opt_alpha, opt_theta, y,
                        model_type, nmse, distribution);
}
```

In `init()` (theta.cpp:177-193) add the enum + binding:

```cpp
  py::enum_<dist::Distribution>(theta, "Distribution")
      .value("Normal", dist::Distribution::Normal)
      .value("Laplace", dist::Distribution::Laplace)
      .value("StudentT", dist::Distribution::StudentT)
      .value("SkewNormal", dist::Distribution::SkewNormal)
      .value("GED", dist::Distribution::GED);
  theta.def("optimize_dist", &optimize_dist);
```

- [ ] **Step 2: Build and smoke-test**

Run: `uv run python setup.py build_ext --inplace && uv run python -c "from statsforecast._lib import theta as t; print(t.Distribution.GED, hasattr(t, 'optimize_dist'))"`
Expected: prints `Distribution.GED True`.

- [ ] **Step 3: Write the failing Theta fitting test**

Append to `tests/test_distributions.py`:

```python
@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_theta_distribution_keys(distribution):
    from statsforecast.theta import thetamodel
    m = thetamodel(ap, m=12, modeltype="STM", initial_smoothed=ap[0] / 2,
                   alpha=0.5, theta=2.0, nmse=3, distribution=distribution)
    assert m["distribution"] == distribution
    if distribution != "normal":
        assert np.isfinite(m["loglik"]) and np.isfinite(m["aic"])
    assert ("nu" in m) == (distribution == "t")
    assert ("alpha_dist" in m) == (distribution == "skew-normal")
    assert ("beta_dist" in m) == (distribution == "ged")
```

- [ ] **Step 4: Run to verify it fails**

Run: `uv run python -m pytest tests/test_distributions.py -q -k theta`
Expected: FAIL with `TypeError: thetamodel() got an unexpected keyword argument 'distribution'`.

- [ ] **Step 5: Thread `distribution` through `theta.py`**

Add `from .distributions import (switch_distribution, dist_init_params, extract_dist_params, aic_bic_aicc)`. In `thetamodel` (theta.py:131) add `distribution="normal"`, always store `distribution` in the returned dict. The optimization becomes an `if/else` that REPLACES (not augments) the existing `optimize_theta_target_fn` call — the normal branch is the current code UNCHANGED (decision D), and the non-normal branch is one-pass, seeded from the SAME initial `par` values `initparamtheta` produces (no normal warm-start, per decision B). The non-normal branch (`par` still holds the INITIAL values here — no prior normal fit):

```python
    if distribution != "normal":
        struct_x0 = np.array(
            [par[k] for k, v in optimize_params.items() if v], dtype=np.float64
        )
        n_struct = len(struct_x0)
        var_init = max(float(np.nanvar(y)), 1e-10)
        n_dist, dist_init = dist_init_params(distribution, var_init)
        # structural bounds (same as optimize_theta_target_fn)
        lb = {"initial_smoothed": -1e10, "alpha": 0.1, "theta": 1.0}
        ub = {"initial_smoothed": 1e10, "alpha": 0.99, "theta": 1e10}
        lower = [lb[k] for k, v in optimize_params.items() if v]
        upper = [ub[k] for k, v in optimize_params.items() if v]
        x0_ext = np.concatenate([struct_x0, dist_init])
        lower_ext = np.concatenate([lower, np.full(n_dist, -np.inf)])
        upper_ext = np.concatenate([upper, np.full(n_dist, np.inf)])
        opt_res = _theta.optimize_dist(
            x0_ext, lower_ext, upper_ext,
            par["initial_smoothed"], par["alpha"], par["theta"],
            optimize_params["initial_smoothed"], optimize_params["alpha"],
            optimize_params["theta"], y, model_type, nmse,
            switch_distribution(distribution, _theta),
        )
        fred = results(*opt_res)
        fit_par = fred.x[:n_struct] if n_dist > 0 else fred.x
        fit_par_dist = fred.x[-n_dist:] if n_dist > 0 else np.array([])
        jj = 0
        for key in ("initial_smoothed", "alpha", "theta"):
            if optimize_params[key]:
                par[key] = fit_par[jj]; jj += 1
```

Then recompute residuals via `_theta.pegels_resid(...)` (already at theta.py:174) with the updated `par`, and for the non-normal case compute the IC using `n_eff = n - 3` and `e[3:]`:

```python
    if distribution != "normal":
        n_eff = len(y) - 3
        n_free = sum(optimize_params.values())
        n_dist_count = 1 if distribution == "laplace" else 2
        np_eff = n_free + n_dist_count
        if distribution == "laplace":
            neg2logL = 2 * n_eff * fred.fn + n_eff * (2 + np.log(4))
        else:
            neg2logL = 2 * n_eff * fred.fn
        aic, bic, aicc = aic_bic_aicc(neg2logL, np_eff, n_eff)
        loglik = -0.5 * neg2logL
        dist_extra = extract_dist_params(distribution, fit_par_dist, residuals=e[3:])
        # add aic/bic/aicc/loglik/sigma2/distribution + shape keys to the returned dict
```

Add `aic`, `bic`, `aicc`, `loglik`, `sigma2`, `distribution`, and the shape key to the returned dict (the normal path keeps its existing `mse`-only dict plus `distribution="normal"`).

In `auto_theta` (theta.py:380) add `distribution="normal"`, thread it into each `thetamodel(...)` call, and switch the selection key:

```python
    ic_key = "mse" if distribution == "normal" else "aic"
    ...
    fit_ic = fit[ic_key]
```

In `forecast_theta` (theta.py:328) pass the fitted distribution to `compute_pi_samples`:

```python
    from statsforecast.distributions import error_params_from_model
    dist = obj.get("distribution", "normal")
    pi = compute_pi_samples(..., error_distribution=dist,
                            error_params=error_params_from_model(obj),
                            residuals=obj["residuals"][3:])
```

- [ ] **Step 6: Thread `distribution` through `AutoTheta` in `models.py`**

Add `distribution: str = "normal"` to `AutoTheta.__init__` (models.py:~1255 region), validate, store, and pass to `auto_theta(...)` in `fit`.

- [ ] **Step 7: Run Theta tests + interval ordering**

Append:

```python
@pytest.mark.parametrize("distribution", ["normal", "laplace", "t", "skew-normal", "ged"])
def test_theta_interval_ordering(distribution):
    from statsforecast.models import AutoTheta
    model = AutoTheta(season_length=12, model="STM", distribution=distribution)
    model.fit(ap)
    assert model.model_["distribution"] == distribution
    pred = model.predict(h=12, level=[80, 95])
    assert np.all(pred["lo-95"] < pred["lo-80"])
    assert np.all(pred["lo-80"] < pred["mean"])
    assert np.all(pred["mean"] < pred["hi-80"])
    assert np.all(pred["hi-80"] < pred["hi-95"])
```

Run: `uv run python -m pytest tests/test_distributions.py tests/test_theta.py -q -k "theta or Theta"`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/theta.cpp python/statsforecast/theta.py python/statsforecast/models.py tests/test_distributions.py
git commit -m "feat(theta): C++ likelihood-based distribution fitting (non-normal) + AIC selection"
```

---

## Task 7: Simulation option-B default + mismatch warning

**Files:**
- Modify: `python/statsforecast/simulation.py` (deprecated `SUPPORTED_DISTRIBUTIONS` alias; import `Distribution`)
- Modify: `python/statsforecast/models.py` (`AutoETS`/`AutoCES`/`AutoTheta` `.simulate` default to fitted distribution + warn on mismatch)
- Test: `tests/test_distributions.py`

**Interfaces:**
- Consumes: `VALID_DISTRIBUTIONS` (Task 2), each model's `model_["distribution"]`.
- Behavior (decision B/option-B + G): `.simulate(error_distribution=None)` defaults to the fitted `model_["distribution"]`; if the caller passes an explicit `error_distribution` that differs, emit a `UserWarning` (does not block). `SUPPORTED_DISTRIBUTIONS` stays importable as a deprecated alias.

- [ ] **Step 1: Write the failing warning test**

Append to `tests/test_distributions.py` (mirrors `tests/test_simulation_warning.py`; this file must NOT set `warnings.simplefilter("ignore")`):

```python
import warnings


def test_simulate_distribution_mismatch_warns():
    from statsforecast.models import AutoETS
    model = AutoETS(season_length=12, model="ANN", distribution="t")
    model.fit(ap)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.simulate(h=4, n_paths=10, error_distribution="laplace")
        assert any("distribution" in str(wi.message).lower() for wi in w)


def test_simulate_defaults_to_fitted_distribution():
    from statsforecast.models import AutoETS
    model = AutoETS(season_length=12, model="ANN", distribution="t")
    model.fit(ap)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.simulate(h=4, n_paths=10)  # no explicit error_distribution
        assert not any("distribution" in str(wi.message).lower() for wi in w)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/test_distributions.py -q -k "mismatch or defaults_to_fitted"`
Expected: FAIL (no warning emitted; `error_distribution` currently defaults to `"normal"`, not the fitted distribution).

- [ ] **Step 3: Add the deprecated alias in `simulation.py`**

Replace the exported frozenset with an internal validation set plus a deprecated alias, and import the shared enum:

```python
from statsforecast.distributions import VALID_DISTRIBUTIONS

_SUPPORTED_SIMULATION_DISTRIBUTIONS = {str(d) for d in VALID_DISTRIBUTIONS} | {"bootstrap"}
# Deprecated public alias (kept for backwards compatibility).
SUPPORTED_DISTRIBUTIONS = frozenset(_SUPPORTED_SIMULATION_DISTRIBUTIONS)
```

Point `_validate_distribution` at `_SUPPORTED_SIMULATION_DISTRIBUTIONS`. Keep `SUPPORTED_DISTRIBUTIONS` in `__all__`.

- [ ] **Step 4: Add option-B logic to the model `.simulate` methods**

In `AutoETS.simulate`, `AutoCES.simulate`, and `AutoTheta.simulate`, change the signature default to `error_distribution: Optional[str] = None` and add at the top of each (after the model dict `mod` is resolved):

```python
        fitted_dist = mod.get("distribution", "normal")
        if error_distribution is None:
            error_distribution = fitted_dist
        elif error_distribution != fitted_dist:
            import warnings
            warnings.warn(
                f"Simulating with error_distribution={error_distribution!r} but the "
                f"model was fitted with distribution={fitted_dist!r}; sigma2 was "
                f"optimised under the fitted distribution.",
                UserWarning,
            )
```

- [ ] **Step 5: Run the warning tests + the simulation suite**

Run: `uv run python -m pytest tests/test_distributions.py tests/test_simulation.py tests/test_simulation_distributions.py tests/test_simulation_warning.py -q`
Expected: PASS (existing simulation tests unaffected; new warning tests pass).

- [ ] **Step 6: Commit**

```bash
git add python/statsforecast/simulation.py python/statsforecast/models.py tests/test_distributions.py
git commit -m "feat(simulate): default error_distribution to fitted; warn on mismatch (option B)"
```

---

## Task 8: Unified parametrized test suite (consolidation)

**Files:**
- Modify: `tests/test_distributions.py` (add the cross-method matrix + scipy loglik check)

**Interfaces:**
- Consumes everything from Tasks 2–7. This task only adds tests; no source changes.

- [ ] **Step 1: Add the cross-method matrix**

Append a parametrization over the models that now support distributions, asserting the model-dict key contract uniformly:

```python
NON_NORMAL = ["laplace", "t", "skew-normal", "ged"]

MODEL_BUILDERS = {
    "AutoARIMA": lambda d: ("statsforecast.models", "AutoARIMA",
                            dict(season_length=12, distribution=d)),
    "AutoETS": lambda d: ("statsforecast.models", "AutoETS",
                          dict(season_length=12, model="ANN", distribution=d)),
    "AutoCES": lambda d: ("statsforecast.models", "AutoCES",
                          dict(season_length=12, model="S", distribution=d)),
    "AutoTheta": lambda d: ("statsforecast.models", "AutoTheta",
                            dict(season_length=12, model="STM", distribution=d)),
}


@pytest.mark.parametrize("model_name", list(MODEL_BUILDERS))
@pytest.mark.parametrize("distribution", ["normal"] + NON_NORMAL)
def test_cross_method_distribution_keys(model_name, distribution):
    import importlib
    mod_name, cls_name, kwargs = MODEL_BUILDERS[model_name](distribution)
    Model = getattr(importlib.import_module(mod_name), cls_name)
    model = Model(**kwargs)
    model.fit(ap)
    md = model.model_
    assert md.get("distribution", "normal") == distribution
    assert ("nu" in md) == (distribution == "t")
    assert ("alpha_dist" in md) == (distribution == "skew-normal")
    assert ("beta_dist" in md) == (distribution == "ged")
```

(If `AutoARIMA`'s fitted dict key differs from `model_`, adjust the accessor for that model only — ARIMA stores its fitted result under `model_.model_`; confirm and use the right attribute.)

- [ ] **Step 2: Add the scipy log-likelihood cross-check**

For models with a full `-2logL` (ETS, ARIMA), recompute the stored `loglik` from the residuals via scipy and assert agreement; for CES/Theta (whose objectives are on a per-model scale), assert only that `loglik` is finite and that the matched distribution improves AIC over normal *within the same model on heavy-tailed data*:

```python
def test_ets_t_aic_better_than_normal_heavy_tails():
    from statsforecast.ets import ets_f
    rng = np.random.default_rng(42)
    e = stats.t.rvs(df=5, size=300, random_state=rng)
    y = np.zeros(300); y[0] = e[0]
    for i in range(1, 300):
        y[i] = 0.8 * y[i - 1] + e[i]
    y = y - y.min() + 1.0
    assert ets_f(y, m=1, model="ANN", distribution="t")["aic"] < \
           ets_f(y, m=1, model="ANN", distribution="normal")["aic"]
```

- [ ] **Step 3: Run the full distribution suite**

Run: `uv run python -m pytest tests/test_distributions.py -q`
Expected: PASS.

- [ ] **Step 4: Run the full project test suite (no regressions)**

Run: `uv run python -m pytest tests/ -q`
Expected: PASS (coverage gate aside).

- [ ] **Step 5: Commit**

```bash
git add tests/test_distributions.py
git commit -m "test: unified parametrized distribution suite across ARIMA/ETS/CES/Theta"
```

---

## Done criteria

- `include/statsforecast/distributions.h` and `python/statsforecast/distributions.py` are the single C++/Python homes for distribution math and fitting helpers.
- ETS and ARIMA import from the shared layers with no behavior change; CES and Theta fit `normal/laplace/t/skew-normal/ged` and expose `distribution=`.
- Prediction intervals reflect the fitted distribution for all four models.
- `.simulate` defaults to the fitted distribution and warns on explicit mismatch.
- `tests/test_distributions.py` is green and the full suite has no regressions.
