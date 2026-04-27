import numpy as np
from statsforecast.utils import _seasonal_naive


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
    """Regression test for #1140: SeasonalNaive with len(y) < season_length."""
    import numpy as np
    from statsforecast.utils import _seasonal_naive
    import warnings
    
    # 4 observations, 12-period season
    y = np.array([1.0, 2.0, 3.0, 4.0])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _seasonal_naive(y=y, h=24, season_length=12, fitted=False)
    
    fcst = result["mean"]
    
    # Positions 0-7 should be NaN (no data for those seasonal positions)
    assert np.isnan(fcst[0])
    assert np.isnan(fcst[7])
    
    # Position 8-11 should be [1,2,3,4] (aligned correctly)
    assert np.isclose(fcst[8], 1.0)
    assert np.isclose(fcst[9], 2.0)
    assert np.isclose(fcst[10], 3.0)
    assert np.isclose(fcst[11], 4.0)
    
    # Warning should be emitted
    assert len(w) >= 1, "No warning emitted for partial season"
    assert "shorter than season_length" in str(w[0].message)
