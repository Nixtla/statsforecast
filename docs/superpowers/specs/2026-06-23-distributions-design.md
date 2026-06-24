# Unified Error Distributions Across the StatsForecast Suite

**Date:** 2026-06-23
**Branch:** `feat/ets_likelihood` (expanded scope)
**Status:** Approved

## Overview

Extend non-normal error distribution fitting — already implemented for ARIMA (PR #1113) and ETS (PR #1116, in progress) — to AutoCES and AutoTheta, and establish shared infrastructure so every future method addition is a single-recipe extension.

The supported distributions are: `normal`, `t`, `laplace`, `skew-normal`, `ged`.

## Motivation

PRs #1113 and #1116 implement distribution-aware MLE fitting independently per method. Without shared infrastructure the C++ distribution math will be duplicated across `ets.cpp`, `ces.cpp`, `theta.cpp` (and more in the future), and the Python distribution utilities are split between `utils.py` and `simulation.py` with an inconsistent naming convention.

## Scope

**In scope for this change:**
- Shared C++ distribution header (`include/statsforecast/distributions.h`)
- Shared Python distribution module (`python/statsforecast/distributions.py`)
- Refactor ETS branch to use both shared layers (cleanup of PR #1116)
- Refactor ARIMA imports to use `distributions.py`
- Distribution-aware MLE fitting for AutoCES and AutoTheta
- Unified parametrized test suite (`tests/test_distributions.py`)

**Out of scope (future PRs):**
- AutoMFLES, AutoTBATS, GARCH
- Adding new distributions beyond the five above

## Architecture

```
include/statsforecast/distributions.h    ← C++ single source of truth
    Distribution enum
    Inline negloglik_*() functions

src/ets.cpp     ← #include header, remove local enum/functions
src/ces.cpp     ← #include header, add optimize_dist()
src/theta.cpp   ← #include header, add optimize_dist()

python/statsforecast/distributions.py   ← Python single source of truth
python/statsforecast/arima.py           ← import from distributions.py
python/statsforecast/ets.py             ← import from distributions.py
python/statsforecast/ces.py             ← distribution= param, import from distributions.py
python/statsforecast/theta.py           ← distribution= param, import from distributions.py
python/statsforecast/utils.py           ← re-exports for backwards compat
python/statsforecast/simulation.py      ← keeps sample_errors(); imports Distribution enum

tests/test_distributions.py            ← unified parametrized test suite
```

Fitting flow for any method when `distribution != "normal"`:
1. Python calls C++ `optimize_dist(x0_model_init, y, ..., distribution)` — a single call that handles both passes internally:
   - **Pass 1 (C++):** runs the existing normal optimizer to get fitted model params and residuals
   - **Pass 2 (C++):** initializes distribution params from residuals (`var(e)` for sigma2, etc.) via `distribution_init_params()` from the shared header; builds extended `x0 = [pass1_model_params..., dist_init...]`; runs Nelder-Mead jointly over model + distribution params under the non-normal likelihood
2. Python extracts distribution params from the result tail via `_distribution_extract_params()` and stores them in the model dict (`nu`, `alpha_dist`, `beta_dist`)
3. Python prediction intervals: `_calculate_intervals()` or `sample_errors()` from `distributions.py` / `simulation.py` using stored params

## C++ Layer — `include/statsforecast/distributions.h`

Single header, included by all method `.cpp` files.

```cpp
#pragma once
#include <cmath>
#include <numbers>
#include <limits>

enum class Distribution { Normal=0, Laplace=1, StudentT=2, SkewNormal=3, GED=4 };

// Returns number of extra optimizer params for the distribution.
// Laplace: 0 (b_hat estimated analytically)
// All others: 2 (log_sigma2 + shape param)
inline int distribution_n_extra_params(Distribution d) {
    return (d == Distribution::Laplace) ? 0 : 2;
}

// Per-observation negative log-likelihood contributions.
// params_tail: the distribution-specific suffix of the optimizer vector.
inline double negloglik_laplace(const double* e, int n);
inline double negloglik_t(const double* e, int n, double log_sigma2, double log_nu_m2);
inline double negloglik_skewnorm(const double* e, int n, double log_sigma2, double alpha);
inline double negloglik_ged(const double* e, int n, double log_sigma2, double log_beta);

// Initialise the distribution-param suffix of x0 from pass-1 residuals.
// Called inside optimize_dist() in each .cpp — not from Python.
inline VectorXd distribution_init_params(const VectorXd& e, Distribution d);
```

**Two-pass structure inside `optimize_dist()` in each `.cpp`:**
```
optimize_dist(x0_model_init, y, ..., distribution):
    1. Call existing optimize(x0_model_init, y, ...)  → normal_params, residuals e
    2. dist_x0 = distribution_init_params(e, distribution)  // from header
    3. x0_ext  = concat(normal_params, dist_x0)
    4. Run Nelder-Mead with ObjectiveFunctionDist over x0_ext
    5. Return extended result vector
```

**Optimizer vector layout (consistent across all methods):**

| Distribution | Extra params appended to x0 |
|---|---|
| normal | — |
| laplace | — (b_hat analytical) |
| t | `[log_sigma2, log(nu−2)]` |
| skew-normal | `[log_sigma2, alpha]` |
| ged | `[log_sigma2, log_beta]` |

Each `.cpp` file:
1. `#include "statsforecast/distributions.h"`
2. Implements `ObjectiveFunctionDist()` — calls method's residual function, dispatches to `negloglik_*`
3. Implements `optimize_dist(x0, y, ..., distribution)` — warm-starts from existing `optimize()`, then runs Nelder-Mead with `ObjectiveFunctionDist`
4. Exposes `Distribution` enum and `optimize_dist` via pybind11

## Python Layer — `distributions.py`

Single module, imported by all method Python files.

```python
class Distribution(str, Enum):
    NORMAL      = "normal"
    LAPLACE     = "laplace"
    T           = "t"
    SKEW_NORMAL = "skew-normal"
    GED         = "ged"

VALID_DISTRIBUTIONS = tuple(Distribution)

def distribution_n_extra_params(distribution: str) -> int: ...
    # mirrors C++ header; used for AIC/BIC penalty

def switch_distribution(d: str, module) -> Any: ...
    # maps string → C++ enum for any module (_ets, _ces, _theta)
    # e.g. switch_distribution("t", _ces) → _ces.Distribution.StudentT

def _quantiles(level, distribution="normal", dist_params=None) -> np.ndarray: ...
    # deterministic PPF via scipy; no RNG

def _calculate_intervals(out, level, h, sigmah, distribution="normal", dist_params=None) -> dict: ...

def _distribution_extract_params(distribution: str, fit_par_dist: np.ndarray) -> dict: ...
    # converts optimizer tail → model dict keys
    # e.g. t → {"nu": exp(fit_par_dist[1]) + 2.0}

def _distribution_aic_correction(distribution: str, n: int, np_base: int,
                                  neg2logL: float) -> tuple[float, float, float]: ...
    # returns (aic, bic, aicc) with correct penalty for distribution extra params
```

**Backwards compatibility:**
- `utils.py` re-exports `Distribution`, `_VALID_DISTRIBUTIONS`, `_quantiles`, `_calculate_intervals` from `distributions.py`
- `simulation.py` keeps `sample_errors()` and its private `_fit_*_distribution()` helpers; imports `Distribution` from `distributions.py` instead of defining its own `SUPPORTED_DISTRIBUTIONS` frozenset. `bootstrap` stays in `simulation.py` (simulation-only, not a fitting distribution). `simulation.py` defines its own validation set internally: `_SUPPORTED_SIMULATION_DISTRIBUTIONS = set(VALID_DISTRIBUTIONS) | {"bootstrap"}` — this replaces the old `SUPPORTED_DISTRIBUTIONS` frozenset without exposing `bootstrap` as a fitting distribution.

`switch_distribution()` consolidates the per-method helpers that currently exist in `arima.py` and `ets.py`. Takes the target module as argument so one function serves all methods.

## Method Integration — CES and Theta

The same recipe applies to both:

**C++ (`ces.cpp`, `theta.cpp`):**
1. Include `distributions.h`
2. Add `ObjectiveFunctionDist(params, y, ..., distribution)` wrapping the method's existing residual function
3. Add `optimize_dist(x0, y, ..., distribution)`:
   - Runs existing `optimize()` to get warm-start model params
   - Appends `_distribution_init_params`-equivalent values to `x0`
   - Runs Nelder-Mead with `ObjectiveFunctionDist`
4. Expose via pybind11

**Python (`ces.py`, `theta.py`):**
1. Add `distribution="normal"` to `cesmodel()` / `thetamodel()`
2. Normal path: unchanged
3. Non-normal path:
   - Call `_ces.optimize_dist(x0_model_init, y, ..., distribution)` — C++ handles both passes internally
   - Extract distribution params from result tail with `_distribution_extract_params()`
   - Compute AIC/BIC/AICc with `_distribution_aic_correction()`
   - Store params in model dict
4. Propagate `distribution=` through `ces_f()` / `theta_f()` to model class

**ETS branch cleanup:** Remove local `Distribution` enum and `negloglik_*` from `ets.cpp`; include the shared header. Replace per-file `switch_distribution()` in `ets.py` with the one from `distributions.py`.

## Fitting vs Simulation Distribution

`distribution=` (fitting) and `error_distribution=` (simulation) are independent parameters. The `simulate()` method:

1. Defaults `error_distribution` to `model.get("distribution", "normal")` — sensible default, no mismatch by default
2. If the caller explicitly passes a different `error_distribution`, emits a `UserWarning` that `sigma2` was optimised under a different distribution
3. Always proceeds — the mismatch is surfaced but not blocked

`sample_errors()` in `simulation.py` handles the actual sampling using stored distribution params from the model dict.

## Testing — `tests/test_distributions.py`

Parametrization is driven entirely from `distributions.VALID_DISTRIBUTIONS` so adding a new distribution automatically covers all models and tests.

```python
MODELS = [AutoARIMA, AutoETS, AutoCES, AutoTheta]
NON_NORMAL = [d for d in VALID_DISTRIBUTIONS if d != Distribution.NORMAL]
```

**Cross-method behaviour tests:**
- `test_fit_returns_loglik` — fitted model dict has `loglik`, `aic`, `bic`, `aicc` for every (Model, distribution)
- `test_distribution_params_stored` — `nu` / `alpha_dist` / `beta_dist` present for non-normal (Model, distribution)
- `test_predict_intervals_finite` — no NaN/inf in prediction intervals for every (Model, distribution)
- `test_simulate_distribution_mismatch_warns` — fitting with normal + simulating with t emits `UserWarning`
- `test_aic_penalizes_extra_params` — AIC penalty count matches `distribution_n_extra_params()` per distribution

**Math correctness vs scipy:**

*Python utilities (deterministic PPF, exact match):*
```python
@pytest.mark.parametrize("distribution,scipy_dist,params", [
    (Distribution.T,          stats.t,        {"df": 5}),
    (Distribution.LAPLACE,    stats.laplace,  {}),
    (Distribution.SKEW_NORMAL, stats.skewnorm, {"a": 1.5}),
    (Distribution.GED,        stats.gennorm,  {"beta": 1.2}),
])
def test_quantiles_match_scipy(distribution, scipy_dist, params):
    # _quantiles() must match scipy PPF at the same probability levels
```

*C++ log-likelihoods (indirect, via stored loglik):*
```python
@pytest.mark.parametrize("Model", MODELS)
@pytest.mark.parametrize("distribution", NON_NORMAL)
def test_loglik_matches_scipy(Model, distribution):
    # 1. Fit model with given distribution
    # 2. Extract stored residuals + fitted dist params (nu, alpha_dist, etc.)
    # 3. Recompute log-likelihood using scipy.stats.*logpdf on those residuals
    # 4. Assert stored loglik matches within 1e-4
```

Existing test files (`test_arima.py` etc.) keep their regression and edge-case tests. `test_distributions.py` owns all cross-method distribution behaviour.
