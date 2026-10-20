import warnings

import numpy as np
import pytest
from statsforecast.utils import _conformal_quantile, _conformal_rank, _seasonal_naive


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
