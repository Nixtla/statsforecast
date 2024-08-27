import pandas as pd
import pytest

from statsforecast import StatsForecast
from statsforecast.utils import AirPassengersDF as df
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoCES,
    AutoTheta,
    MSTL,
    GARCH,
    ARCH,
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
    WindowAverage,
    SeasonalWindowAverage,
    ADIDA,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    IMAPA,
    TSB,
)


def get_data(rename_cols: bool = False):
    df2 = df.copy(deep=True)
    df2["unique_id"] = 2.0
    df2["y"] *= 2
    panel_df = pd.concat([df, df2])[["unique_id", "ds", "y"]]
    if rename_cols:
        renamer = {
            "unique_id": "item_id",
            "ds": "timestamp",
            "y": "target",
        }
        panel_df.rename(columns=renamer, inplace=True)
    return panel_df


@pytest.mark.parametrize("n_jobs", [-1, 1])
def test_custom_cols(n_jobs):
    AirPassengersPanel = get_data(rename_cols=True)
    custom_cols = {
        "id_col": "item_id",
        "time_col": "timestamp",
        "target_col": "target",
    }
    models = [
        HistoricAverage(),
        Naive(),
        RandomWalkWithDrift(),
        SeasonalNaive(season_length=12),
    ]
    sf = StatsForecast(models=models, freq="D", n_jobs=n_jobs)
    df_fcst = sf.forecast(df=AirPassengersPanel, h=7, fitted=True, **custom_cols)
    assert all(
        exp_col in df_fcst.columns
        for exp_col in custom_cols.values()
        if exp_col != "target"
    )
    assert all(
        exp_col in sf.forecast_fitted_values().columns
        for exp_col in custom_cols.values()
    )
    sf.fit(df=AirPassengersPanel, **custom_cols)
    df_predict = sf.predict(h=7)
    pd.testing.assert_frame_equal(df_fcst, df_predict)


@pytest.mark.parametrize("n_jobs", [-1, 1])
def test_fit_predict(n_jobs):
    AirPassengersPanel = get_data()
    models = [
        AutoARIMA(),
        AutoETS(),
        AutoCES(),
        AutoTheta(),
        MSTL(season_length=12),
        GARCH(),
        ARCH(),
        HistoricAverage(),
        Naive(),
        RandomWalkWithDrift(),
        SeasonalNaive(season_length=12),
        WindowAverage(window_size=12),
        SeasonalWindowAverage(window_size=12, season_length=12),
        ADIDA(),
        CrostonClassic(),
        CrostonOptimized(),
        CrostonSBA(),
        IMAPA(),
        TSB(0.5, 0.5),
    ]

    sf = StatsForecast(models=models, freq="D", n_jobs=n_jobs)
    df_fcst = sf.forecast(df=AirPassengersPanel, h=7)
    sf.fit(df=AirPassengersPanel)
    df_predict = sf.predict(h=7)
    pd.testing.assert_frame_equal(df_fcst, df_predict)
