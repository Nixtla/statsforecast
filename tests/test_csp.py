import warnings

import numpy as np
import pytest

from statsforecast.models import ConformalSeasonalPool, SeasonalNaive


@pytest.fixture
def y_monthly():
    rng = np.random.default_rng(42)
    return (rng.standard_normal(60).cumsum() + 100).astype(np.float32)


@pytest.fixture
def y_short():
    """Series with fewer than 3 full seasonal cycles (2 cycles = 24 obs for m=12)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(24).cumsum() + 100).astype(np.float32)


# 4.1 — Smoke test
def test_forecast_keys_and_shapes(y_monthly):
    res = ConformalSeasonalPool(season_length=12).forecast(
        y_monthly, h=12, level=[80, 95]
    )
    assert set(res.keys()) == {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
    for v in res.values():
        assert v.shape == (12,)


# 4.2 — Point forecast matches SeasonalNaive exactly
def test_mean_matches_seasonal_naive(y_monthly):
    csp = ConformalSeasonalPool(season_length=12).forecast(y_monthly, h=12)
    sn = SeasonalNaive(season_length=12).forecast(y_monthly, h=12)
    np.testing.assert_array_equal(csp["mean"], sn["mean"])


# 4.3 — Interval ordering: lo-95 <= lo-80 <= (no guarantee vs mean, sampling) <= hi-80 <= hi-95
def test_interval_ordering(y_monthly):
    res = ConformalSeasonalPool(season_length=12, n_samples=500).forecast(
        y_monthly, h=12, level=[80, 95]
    )
    assert np.all(res["lo-95"] <= res["lo-80"] + 1e-5)
    assert np.all(res["hi-80"] <= res["hi-95"] + 1e-5)


# 4.4 — season_length=1: w=0, pure residual sampler; mean equals naive forecast
def test_no_seasonality_w0():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(50).cumsum() + 50
    res = ConformalSeasonalPool(season_length=1, n_samples=200).forecast(
        y, h=6, level=[90]
    )
    # Mean = naive (last value repeated)
    np.testing.assert_array_almost_equal(res["mean"], np.full(6, y[-1]))


# 4.5 — Adaptive variant on short series (< 3 cycles) triggers w=0.3
def test_thin_history_adaptive(y_short):
    # y_short has exactly 24 obs = 2 full cycles with m=12  →  n//m = 2 < 3
    csp = ConformalSeasonalPool(season_length=12, variant="adaptive")
    n, m = y_short.size, 12
    assert n // m < 3  # guard: confirms fixture is short enough
    # Should produce valid output without error
    res = csp.forecast(y_short, h=12, level=[80])
    assert "lo-80" in res


# 4.6 — Variant comparison: fixed and adaptive differ on short series
def test_variant_comparison_short_series(monkeypatch, y_short):
    # Seed both calls identically so the only difference is the mixture weight logic.
    rngs = iter([np.random.default_rng(0), np.random.default_rng(0)])
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **kw: next(rngs))
    fixed = ConformalSeasonalPool(
        season_length=12, variant="fixed", n_samples=500
    ).forecast(y_short, h=12, level=[80])
    adaptive = ConformalSeasonalPool(
        season_length=12, variant="adaptive", n_samples=500
    ).forecast(y_short, h=12, level=[80])
    # Fixed uses w=0.5 always; adaptive uses w=0.3 here → different interval widths
    fixed_width = fixed["hi-80"] - fixed["lo-80"]
    adaptive_width = adaptive["hi-80"] - adaptive["lo-80"]
    # Not identical (different mixture weights lead to different distributions)
    assert not np.allclose(fixed_width, adaptive_width, atol=1e-3)


# 4.7 — Simulate seed reproducibility
def test_simulate_reproducibility(y_monthly):
    model = ConformalSeasonalPool(season_length=12)
    a = model.simulate(h=12, n_paths=100, y=y_monthly, seed=42)
    b = model.simulate(h=12, n_paths=100, y=y_monthly, seed=42)
    np.testing.assert_array_equal(a, b)


# 4.8 — Simulate shape
def test_simulate_shape(y_monthly):
    model = ConformalSeasonalPool(season_length=12)
    paths = model.simulate(h=6, n_paths=50, y=y_monthly)
    assert paths.shape == (50, 6)


# 4.9 — Simulate warning on non-default error_distribution
def test_simulate_warns_on_error_distribution(y_monthly):
    model = ConformalSeasonalPool(season_length=12)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.simulate(h=6, n_paths=10, y=y_monthly, error_distribution="laplace")
    assert any(issubclass(warning.category, UserWarning) for warning in w)
    assert any("CSP" in str(warning.message) for warning in w)


# 4.10 — predict_in_sample structure and constant-width intervals
def test_predict_in_sample(y_monthly):
    model = ConformalSeasonalPool(season_length=12).fit(y_monthly)
    res = model.predict_in_sample(level=[90])
    assert "fitted" in res
    assert "fitted-lo-90" in res
    assert "fitted-hi-90" in res
    assert res["fitted"].shape == y_monthly.shape
    assert res["fitted-lo-90"].shape == y_monthly.shape
    # Interval width must be constant (ignoring NaN positions)
    width = res["fitted-hi-90"] - res["fitted-lo-90"]
    valid = ~np.isnan(width)
    assert np.allclose(width[valid], width[valid][0])


# 4.11 — fit/predict point forecast matches forecast()
def test_fit_predict_mean_matches_forecast(y_monthly):
    model = ConformalSeasonalPool(season_length=12)
    via_predict = model.fit(y_monthly).predict(h=12, level=[95])
    via_forecast = ConformalSeasonalPool(season_length=12).forecast(
        y_monthly, h=12, level=[95]
    )
    np.testing.assert_array_equal(via_predict["mean"], via_forecast["mean"])


# 4.12 — Horizon beyond one season: mean repeats correctly
def test_horizon_beyond_season_length(y_monthly):
    res = ConformalSeasonalPool(season_length=12).forecast(
        y_monthly, h=24, level=[95]
    )
    assert res["mean"].shape == (24,)
    np.testing.assert_array_equal(res["mean"][12:], res["mean"][:12])


# Bonus: alias auto-derivation
def test_alias_auto_derived():
    assert ConformalSeasonalPool(season_length=12).alias == "CSP-Adaptive"
    assert ConformalSeasonalPool(season_length=12, variant="fixed").alias == "CSP-Fixed"
    assert ConformalSeasonalPool(season_length=12, alias="MyCSP").alias == "MyCSP"


# Bonus: invalid variant raises
def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        ConformalSeasonalPool(season_length=12, variant="bogus")


def test_short_series_no_index_error():
    """n < season_length: forecast must not raise; NaN slots filled via latest-obs fallback."""
    m = 12
    y_short = np.arange(1, 7, dtype=np.float32)  # only 6 obs, m=12 (n <= m/2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ConformalSeasonalPool(season_length=m).forecast(y_short, h=m, level=None)
    mean = result["mean"]
    assert mean.shape == (m,), f"expected shape ({m},), got {mean.shape}"
    assert not np.isnan(mean).any(), "all horizon steps should be non-NaN after latest-obs fallback"


def test_short_series_between_half_and_full_season():
    """n in (m/2, m): residual slice must not crash (negative index bug regression)."""
    m = 12
    y_short = np.arange(1, 9, dtype=np.float32)  # 8 obs, m=12, m/2 < n < m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ConformalSeasonalPool(season_length=m).forecast(y_short, h=m, level=None)
    mean = result["mean"]
    assert mean.shape == (m,), f"expected shape ({m},), got {mean.shape}"
    assert not np.isnan(mean).any(), "all horizon steps should be non-NaN"


def test_empty_R_pool_draws_nondegenerate():
    """R empty but pool k>=2: intervals must have non-zero width from pool draws alone.

    Uses n=2*m with calib_frac≈0 to force R empty while each phase has k=2 observations.
    The old `k==0 or R.size==0` guard collapsed to a point forecast in this case;
    the corrected `k==0 and R.size==0` guard allows pure pool draws instead.
    """
    m = 12
    rng_np = np.random.default_rng(42)
    y = rng_np.standard_normal(2 * m).astype(np.float32)
    # calib_frac=0.01 → t_cal=floor(0.01*24)=0 → R is empty; each phase has k=2 pool obs
    result = ConformalSeasonalPool(
        season_length=m, n_samples=500, calib_frac=0.01
    ).forecast(y, h=m, level=[90])
    lo, hi = result["lo-90"], result["hi-90"]
    assert not np.isnan(lo).any(), "lo intervals should not be NaN"
    assert not np.isnan(hi).any(), "hi intervals should not be NaN"
    assert np.any(hi > lo), "pure pool draws from k=2 pool should produce non-zero width"


def test_predict_in_sample_empty_R_returns_nan_intervals():
    """n == m: fit stores empty R; predict_in_sample(level=...) must not raise."""
    m = 12
    rng = np.random.default_rng(1)
    y = rng.standard_normal(m).astype(np.float32)
    model = ConformalSeasonalPool(season_length=m).fit(y)
    result = model.predict_in_sample(level=[90])
    assert "fitted-lo-90" in result and "fitted-hi-90" in result
    assert np.isnan(result["fitted-lo-90"]).all(), "offsets should be NaN when R is empty"
    assert np.isnan(result["fitted-hi-90"]).all(), "offsets should be NaN when R is empty"
