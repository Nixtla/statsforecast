import numpy as np
import pandas as pd
from statsforecast.mstl import mstl
from statsforecast import StatsForecast
from statsforecast.models import MSTL
from utilsforecast.data import generate_series


def test_mstl_cv_refit_false():
    """Regression test for for CV with refit=False and short series."""
    df = generate_series(n_series=1, freq="MS", min_length=25, max_length=25)
    sf = StatsForecast(
        models=[MSTL(season_length=[12])],
        freq="MS",
        n_jobs=1,
    )
    cv = sf.cross_validation(
        df=df,
        h=12,
        step_size=1,
        n_windows=3,
        refit=False,
    )
    assert len(cv) > 0
    assert "MSTL" in cv.columns


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
    hourly_data = electricity_data.set_index("ds").resample("H").sum()

    seasonal_periods = [24, 24 * 7]  # Daily and weekly patterns
    decomposition = mstl(hourly_data["y"].values, seasonal_periods)

    assert decomposition is not None, "MSTL decomposition should return a result"
    assert hasattr(decomposition, "plot"), "Decomposition should have a plot method"
