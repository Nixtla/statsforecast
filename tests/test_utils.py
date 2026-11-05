import warnings

import numpy as np
import pytest
from statsforecast.utils import (
    _conformal_quantile,
    _conformal_rank,
    _seasonal_naive,
    find_season_length,
)


def test_seasonal_naive():
    # test seasonal naive
    y = np.array([0.50187596, 0.40536128, 0.33436676, 0.27868117, 0.25251294,
              0.18961286, 0.07082107, 2.58699709, 3.06466854, 2.25150509,
              1.33027107, 0.73332616, 0.50187596, 0.40536128, 0.33436676,
              0.27868117, 0.25251294, 0.18961286, 0.07082107, 2.58699709,
              3.06466854, 2.25150509, 1.33027107, 0.73332616, 0.50187596,
              0.40536128, 0.33436676, 0.27868117, 0.25251294, 0.18961286,
              0.07082107, 2.58699709, 3.06466854, 2.25150509, 1.33027107,
              0.73332616, 0.50187596, 0.40536128, 0.33436676, 0.27868117,
              0.25251294, 0.18961286, 0.07082107, 2.58699709, 3.06466854,
              2.25150509, 1.33027107, 0.73332616, 0.50187596, 0.40536128,
              0.33436676, 0.27868117, 0.25251294, 0.18961286, 0.07082107,
              2.58699709, 3.06466854, 2.25150509, 1.33027107, 0.73332616,
              0.50187596, 0.40536128, 0.33436676, 0.27868117, 0.25251294,
              0.18961286, 0.07082107, 2.58699709, 3.06466854, 2.25150509,
              1.33027107, 0.73332616, 0.50187596, 0.40536128, 0.33436676,
              0.27868117, 0.25251294, 0.18961286, 0.07082107, 2.58699709,
              3.06466854, 2.25150509, 1.33027107, 0.73332616, 0.50187596,
              0.40536128, 0.33436676, 0.27868117, 0.25251294, 0.18961286,
              0.07082107, 2.58699709, 3.06466854, 2.25150509, 1.33027107,
              0.73332616, 0.50187596, 0.40536128, 0.33436676, 0.27868117,
              0.25251294, 0.18961286, 0.07082107, 2.58699709, 3.06466854,
              2.25150509, 1.33027107, 0.73332616, 0.50187596, 0.40536128,
              0.33436676, 0.27868117, 0.25251294, 0.18961286, 0.07082107,
              2.58699709, 3.06466854, 2.25150509, 1.33027107, 0.73332616,
              0.50187596, 0.40536128, 0.33436676, 0.27868117, 0.25251294,
              0.18961286])  # fmt: skip

    seas_naive_fcst = dict(_seasonal_naive(y=y, h=12, season_length=12, fitted=True))['mean']  # fmt: skip
    np.testing.assert_array_almost_equal(seas_naive_fcst, y[-12:])

    y = np.array([0.05293832, 0.10395079, 0.25626143, 0.61529232, 1.08816604,
              0.54493457, 0.43415014, 0.47676606, 5.32806397, 3.00553563,
              0.04473598, 0.04920475, 0.05293832, 0.10395079, 0.25626143,
              0.61529232, 1.08816604, 0.54493457, 0.43415014, 0.47676606,
              5.32806397, 3.00553563, 0.04473598, 0.04920475, 0.05293832,
              0.10395079, 0.25626143, 0.61529232, 1.08816604, 0.54493457,
              0.43415014, 0.47676606, 5.32806397, 3.00553563, 0.04473598,
              0.04920475, 0.05293832, 0.10395079, 0.25626143, 0.61529232,
              1.08816604, 0.54493457, 0.43415014, 0.47676606, 5.32806397,
              3.00553563, 0.04473598, 0.04920475, 0.05293832, 0.10395079,
              0.25626143, 0.61529232, 1.08816604])  # fmt: skip
    seas_naive_fcst = dict(_seasonal_naive(y=y, h=12, season_length=12, fitted=True))['mean']  # fmt: skip
    np.testing.assert_array_almost_equal(seas_naive_fcst, y[-12:])

    y = np.arange(23)
    seas_naive_fcst = _seasonal_naive(y, h=12, fitted=True, season_length=12)
    np.testing.assert_equal(
        seas_naive_fcst["fitted"], np.hstack([np.full(12, np.nan), y[:11]])
    )

def test_seasonal_naive_partial_season():
    # test seasonal naive when the number of observations is shorter than the season length 
    y = np.array([1.0, 2.0, 3.0, 4.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _seasonal_naive(y=y, h=24, season_length=12, fitted=False)

    expected = np.array(
        [np.nan] * 8 + [1.0, 2.0, 3.0, 4.0] + [np.nan] * 8 + [1.0, 2.0, 3.0, 4.0]
    )
    np.testing.assert_array_equal(result["mean"], expected)

    assert len(w) >= 1
    assert "shorter than season_length" in str(w[-1].message)


@pytest.mark.parametrize(
    "n, q, upper, expected",
    [
        # upper bound: ceil((n + 1) * q), capped at n
        (2, 0.8, True, 2),
        (9, 0.9, True, 9),
        (19, 0.9, True, 18),
        (20, 0.8, True, 17),
        (99, 0.95, True, 95),
        (100, 0.975, True, 99),
        # an upper bound below the median still rounds up
        (10, 0.4, True, 5),
        (10, 0.3, True, 4),
        # lower bound: floor((n + 1) * q), floored at 1
        (100, 0.025, False, 2),
        (100, 0.05, False, 5),
        (10, 0.1, False, 1),
        (30, 0.025, False, 1),
    ],
)
def test_conformal_rank(n, q, upper, expected):
    assert _conformal_rank(n, q, upper) == expected


def test_conformal_quantile_is_an_exact_order_statistic():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((10, 3))
    srt = np.sort(scores, axis=0)
    # scalar level: the ceil(11 * 0.8) = 9th smallest, no interpolation
    np.testing.assert_array_equal(_conformal_quantile(scores, 0.8, upper=True), srt[8])
    # sequence of levels adds a leading axis, as np.quantile does
    out = _conformal_quantile(scores, [0.1, 0.9], [False, True])
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, srt[[0, 9]])
    # other axes work the same way
    np.testing.assert_array_equal(
        _conformal_quantile(scores.T, 0.8, upper=True, axis=1), srt[8]
    )


@pytest.mark.parametrize("n, level", [(10, 80), (20, 90), (50, 95)])
def test_conformal_quantile_tails_cover(n, level):
    # Each tail of a two-sided interval should miss at most (1 - level) / 2 of
    # new exchangeable draws. Linear interpolation misses the lower tail.
    rng = np.random.default_rng(1)
    reps = 20_000
    alpha = 1 - level / 100
    draws = rng.standard_normal((reps, n + 1))
    lo, hi = _conformal_quantile(
        draws[:, :n], [alpha / 2, 1 - alpha / 2], [False, True], axis=1
    )
    se = np.sqrt(alpha / 2 * (1 - alpha / 2) / reps)
    assert np.mean(draws[:, n] < lo) <= alpha / 2 + 3 * se
    assert np.mean(draws[:, n] > hi) <= alpha / 2 + 3 * se


def test_conformal_quantile_rejects_empty_scores():
    with pytest.raises(ValueError, match="zero scores"):
        _conformal_quantile(np.empty((0, 2)), 0.9, upper=True)


@pytest.mark.parametrize("level", [30, 40, 49])
def test_conformal_quantile_upper_bound_below_the_median_covers(level):
    # An absolute-score margin is an upper bound even when level < 50, so it
    # must round the rank up to cover at least `level` percent.
    rng = np.random.default_rng(2)
    n, reps = 10, 20_000
    scores = np.abs(rng.standard_normal((reps, n + 1)))
    margin = _conformal_quantile(scores[:, :n], level / 100, upper=True, axis=1)
    se = np.sqrt(0.25 / reps)
    assert np.mean(scores[:, n] <= margin) >= level / 100 - 3 * se


def _seasonal(period, n=240, amplitude=5.0, trend=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        amplitude * np.sin(2 * np.pi * t / period) + trend * t + rng.standard_normal(n)
    )


def test_find_season_length_air_passengers(air_passengers):
    assert find_season_length(air_passengers) == 12
    assert find_season_length(np.log(air_passengers)) == 12


@pytest.mark.parametrize("period", [4, 7, 12, 24])
@pytest.mark.parametrize("trend", [0.0, 0.1])
def test_find_season_length_recovers_known_periods(period, trend):
    assert find_season_length(_seasonal(period, trend=trend)) == period


def test_find_season_length_noiseless_sinusoid():
    t = np.arange(120)
    assert find_season_length(np.sin(2 * np.pi * t / 12)) == 12


@pytest.mark.parametrize("scale", [1e-4, 1e-2, 1.0, 1e3, 1e6])
@pytest.mark.parametrize("shift", [0.0, 1e4])
def test_find_season_length_does_not_depend_on_units(air_passengers, scale, shift):
    # R's findfrequency returns 1 for AirPassengers / 100: its peak threshold
    # is applied to the unscaled spectrum.
    assert find_season_length(scale * air_passengers + shift) == 12
    assert find_season_length(scale * _seasonal(7) + shift) == 7


def test_find_season_length_returns_one_without_seasonality():
    rng = np.random.default_rng(1)
    n_series = 300
    white_noise = [rng.standard_normal(120) for _ in range(n_series)]
    random_walks = [np.cumsum(rng.standard_normal(120)) for _ in range(n_series)]
    for series in (white_noise, random_walks):
        detected = sum(find_season_length(y) != 1 for y in series)
        assert detected <= 0.05 * n_series


@pytest.mark.parametrize(
    "y",
    [np.full(50, 3.0), np.arange(50.0), np.array([1.0, 2.0, 1.0]), np.array([])],
)
def test_find_season_length_degenerate_series(y):
    assert find_season_length(y) == 1


def test_find_season_length_uses_longest_finite_stretch():
    y = _seasonal(12)
    with_gaps = y.copy()
    with_gaps[:5] = np.nan
    with_gaps[10] = np.inf
    assert find_season_length(with_gaps) == find_season_length(y[11:]) == 12


def test_find_season_length_max_season_length():
    t = np.arange(480)
    rng = np.random.default_rng(3)
    y = (
        6 * np.sin(2 * np.pi * t / 24)
        + 3 * np.sin(2 * np.pi * t / 6)
        + rng.standard_normal(t.size)
    )
    assert find_season_length(y) == 24
    assert find_season_length(y, max_season_length=12) == 6
    assert find_season_length(y, max_season_length=24) == 24


def test_find_season_length_max_season_length_without_peak_in_band():
    assert find_season_length(_seasonal(24), max_season_length=4) == 1


@pytest.mark.parametrize("bad", [0, 1, -3])
def test_find_season_length_rejects_invalid_max(bad):
    with pytest.raises(ValueError, match="max_season_length"):
        find_season_length(_seasonal(12), max_season_length=bad)


def test_find_season_length_rejects_2d_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        find_season_length(np.ones((10, 2)))

