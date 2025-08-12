import pandas as pd
import polars as pl
import pytest
from statsforecast.feature_engineering import MSTL, mstl_decomposition
from statsforecast.models import Naive
from statsforecast.utils import generate_series
from utilsforecast.losses import smape


@pytest.fixture
def series():
    series = generate_series(10, freq="D")
    series["unique_id"] = series["unique_id"].astype("int64")
    return series


def test_mstl_decomposition(series):
    with pytest.raises(Exception) as exec:
        mstl_decomposition(series, Naive(), "D", 14)
    assert "must be an MSTL instance" in str(exec.value)


def test_mstl_other_params(series):
    horizon = 14
    model = MSTL(season_length=7)
    series = series.sample(frac=1.0)

    train_df, X_df = mstl_decomposition(series, model, "D", horizon)
    series_pl = generate_series(10, freq="D", engine="polars")
    series_pl = series_pl.with_columns(unique_id=pl.col("unique_id").cast(pl.Int64))
    train_df_pl, X_df_pl = mstl_decomposition(series_pl, model, "1d", horizon)
    pd.testing.assert_series_equal(
        train_df.groupby("unique_id")["ds"].max() + pd.offsets.Day(),
        X_df.groupby("unique_id")["ds"].min(),
    )
    assert X_df.shape[0] == train_df["unique_id"].nunique() * horizon
    pd.testing.assert_frame_equal(train_df, train_df_pl.to_pandas())
    pd.testing.assert_frame_equal(X_df, X_df_pl.to_pandas())
    with_estimate = train_df_pl.with_columns(
        estimate=pl.col("trend") + pl.col("seasonal")
    )
    assert smape(with_estimate, models=["estimate"])["estimate"].mean() < 0.1
    model = MSTL(season_length=[7, 28])
    train_df, X_df = mstl_decomposition(series, model, "D", horizon)
    assert train_df.columns.intersection(X_df.columns).tolist() == [
        "unique_id",
        "ds",
        "trend",
        "seasonal7",
        "seasonal28",
    ]
