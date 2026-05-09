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


def test_py_typed_marker_is_packaged():
    """Regression for #1121 — PEP 561 requires a `py.typed` marker file at
    the package root for type checkers (pyright/mypy) to honour the inline
    annotations shipped in this codebase. Without it, importing
    `statsforecast` triggers warnings such as ``Stub file not found for
    "statsforecast"`` in pylance.

    Assert the marker exists *next to* the imported package — the same
    test wheel/install paths a user would hit.
    """
    import os
    import statsforecast

    pkg_dir = os.path.dirname(statsforecast.__file__)
    marker = os.path.join(pkg_dir, "py.typed")
    assert os.path.isfile(marker), (
        f"py.typed marker not found at {marker}; PEP 561 type checkers "
        "won't pick up `statsforecast`'s inline annotations."
    )
