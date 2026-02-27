import numpy as np
import pandas as pd
from statsforecast.mstl import mstl
from statsforecast import StatsForecast
from statsforecast.models import MSTL, Naive
from utilsforecast.data import generate_series


def test_mstl_cv_refit_false_short_windows():
    """MSTL CV with refit=False returns NaNs for early short windows (issue #969)."""
    freq = "MS"
    season_length = 12
    min_length = 25
    df = generate_series(n_series=1, freq=freq, min_length=min_length, max_length=min_length)

    sf = StatsForecast(
        models=[MSTL(season_length=[season_length])],
        freq=freq,
        n_jobs=1,
        fallback_model=Naive(),
    )
    cv_df = sf.cross_validation(
        df=df,
        h=12,
        step_size=1,
        n_windows=3,
        refit=False,
    )
    # First window has 11 train obs (< season_length=12), so it returns NaNs
    # We get 3 windows * 12 steps = 36 rows
    assert len(cv_df) == 36
    assert "MSTL" in cv_df.columns
    assert cv_df["MSTL"].isna().sum() == 12
    assert cv_df["MSTL"].notna().sum() == 24


def test_mstl_cv_refit_false_short_windows_skip():
    """MSTL CV with refit=False skips early short windows when configured."""
    freq = "MS"
    season_length = 12
    min_length = 25
    df = generate_series(n_series=1, freq=freq, min_length=min_length, max_length=min_length)

    sf = StatsForecast(
        models=[MSTL(season_length=[season_length], short_train_behavior="skip")],
        freq=freq,
        n_jobs=1,
        fallback_model=Naive(),
    )
    cv_df = sf.cross_validation(
        df=df,
        h=12,
        step_size=1,
        n_windows=3,
        refit=False,
    )
    # First window has 11 train obs (< season_length=12), so it is skipped
    # We get 2 valid windows * 12 steps = 24 rows (not 36)
    assert len(cv_df) == 24
    assert "MSTL" in cv_df.columns
    assert cv_df["MSTL"].notna().all()


def test_mstl_cv_refit_false_short_windows_nan_single_season():
    """MSTL CV refit=False returns NaNs when initial window is short."""
    freq = "MS"
    season_length = 12
    min_length = 24
    df = generate_series(n_series=1, freq=freq, min_length=min_length, max_length=min_length)

    sf = StatsForecast(
        models=[MSTL(season_length=[season_length], short_train_behavior="nan")],
        freq=freq,
        n_jobs=1,
        fallback_model=Naive(),
    )
    cv_df = sf.cross_validation(
        df=df,
        h=12,
        step_size=1,
        n_windows=2,
        refit=False,
    )
    # First window has 10 train obs (< season_length=12), so it returns NaNs
    assert len(cv_df) == 24
    assert "MSTL" in cv_df.columns
    assert cv_df["MSTL"].isna().sum() == 12
    assert cv_df["MSTL"].notna().sum() == 12


def test_mstl_cv_refit_false_short_windows_skip_single_season():
    """MSTL CV refit=False skips short initial window when configured."""
    freq = "MS"
    season_length = 12
    min_length = 24
    df = generate_series(n_series=1, freq=freq, min_length=min_length, max_length=min_length)

    sf = StatsForecast(
        models=[MSTL(season_length=[season_length], short_train_behavior="skip")],
        freq=freq,
        n_jobs=1,
        fallback_model=Naive(),
    )
    cv_df = sf.cross_validation(
        df=df,
        h=12,
        step_size=1,
        n_windows=2,
        refit=False,
    )
    # First window has 10 train obs (< season_length=12), so it is skipped
    assert len(cv_df) == 12
    assert "MSTL" in cv_df.columns
    assert cv_df["MSTL"].notna().all()


def test_mstl():
    """Test MSTL decomposition with electricity demand data."""

    # Load electricity demand data from online source
    url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
    df = pd.read_csv(url)

    df["Date"] = df["Date"].apply(
        lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
    )
    df["ds"] = df["Date"] + pd.to_timedelta((df["Period"] - 1) * 30, unit="m")

    electricity_data = df[["ds", "OperationalLessIndustrial"]]
    electricity_data.columns = ["ds", "y"]

    # Filter to first 149 days of 2012 for consistent test data
    start_date = pd.to_datetime("2012-01-01")
    end_date = start_date + pd.Timedelta("149D")
    mask = (electricity_data["ds"] >= start_date) & (electricity_data["ds"] < end_date)
    electricity_data = electricity_data[mask]

    # Resample to hourly frequency for analysis
    hourly_data = electricity_data.set_index("ds").resample("h").sum()

    seasonal_periods = [24, 24 * 7]  # Daily and weekly patterns
    decomposition = mstl(hourly_data["y"].values, seasonal_periods)

    assert decomposition is not None, "MSTL decomposition should return a result"
    assert hasattr(decomposition, "plot"), "Decomposition should have a plot method"
