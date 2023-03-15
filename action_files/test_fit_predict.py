import pandas as pd

from neuralforecast.utils import AirPassengersPanel
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoCES, AutoTheta,
    MSTL, GARCH, ARCH, HistoricAverage,
    Naive, RandomWalkWithDrift,
    SeasonalNaive, WindowAverage, SeasonalWindowAverage,
    ADIDA, CrostonClassic,
    CrostonOptimized, CrostonSBA, IMAPA,
    TSB
)


def test_fit_predict_parallel():
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
        IMAPA(),
        TSB(0.5, 0.5)
    ]
        
    sf = StatsForecast(models=models, freq='D', n_jobs=-1)
    df_fcst = sf.forecast(df=AirPassengersPanel[['unique_id', 'ds', 'y']], h=7)
    sf.fit(df=AirPassengersPanel[['unique_id', 'ds', 'y']])
    df_predict = sf.predict(h=7)
    pd.testing.assert_frame_equal(df_fcst, df_predict)


def test_fit_predict_sequential():
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
        IMAPA(),
        TSB(0.5, 0.5)
    ]
        
    sf = StatsForecast(models=models, freq='D')
    df_fcst = sf.forecast(df=AirPassengersPanel[['unique_id', 'ds', 'y']], h=7)
    sf.fit(df=AirPassengersPanel[['unique_id', 'ds', 'y']])
    df_predict = sf.predict(h=7)
    pd.testing.assert_frame_equal(df_fcst, df_predict)
