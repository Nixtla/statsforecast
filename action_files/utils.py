import fugue.api as fa
import pandas as pd

from statsforecast.core import StatsForecast
from statsforecast.models import ( 
    ADIDA,
    AutoARIMA,
    ARIMA,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    AutoETS,
    HistoricAverage,
    IMAPA,
    Naive,
    RandomWalkWithDrift,
    SeasonalExponentialSmoothing,
    SeasonalNaive,
    SeasonalWindowAverage,
    SimpleExponentialSmoothing,
    TSB,
    WindowAverage,
)


def pipeline(series, X_df, n_series, horizon, id_col='unique_id', time_col='ds', target_col='y'):
    models = [
        ADIDA(),
        AutoARIMA(season_length=7),
        ARIMA(season_length=7, order=(0, 1, 2)),
        CrostonClassic(),
        CrostonOptimized(),
        CrostonSBA(),
        AutoETS(season_length=7),
        HistoricAverage(),
        IMAPA(),
        Naive(),
	RandomWalkWithDrift(),
	SeasonalExponentialSmoothing(season_length=7, alpha=0.1),
	SeasonalNaive(season_length=7),
	SeasonalWindowAverage(season_length=7, window_size=4),
	SimpleExponentialSmoothing(alpha=0.1),
	TSB(alpha_d=0.1, alpha_p=0.3),
	WindowAverage(window_size=4)
    ]
    sf = StatsForecast(
        models=models,
        freq='D',
    )
    forecast = fa.as_pandas(
        sf.forecast(df=series, h=horizon, X_df=X_df, id_col=id_col, time_col=time_col, target_col=target_col)
    )
    print(forecast)
    assert forecast.shape == (n_series * horizon, len(models) + 2)

    n_windows = 2
    cv = fa.as_pandas(
        sf.cross_validation(df=series, n_windows=n_windows, h=horizon, id_col=id_col, time_col=time_col, target_col=target_col)
    )
    assert cv.shape[0] == n_series * n_windows * horizon
    assert cv.columns.tolist() == [id_col, time_col, 'cutoff', target_col] + [m.alias for m in models]

def pipeline_with_level(series, X_df, n_series, horizon):
    models = [AutoARIMA(season_length=7)]
    sf = StatsForecast(
        models=models,
        freq='D',
    )
    forecast = fa.as_pandas(sf.forecast(df=series, h=horizon, X_df=X_df, level=[80, 90]))
    print(forecast.columns)
    expected = ["unique_id","ds","AutoARIMA","AutoARIMA-lo-90","AutoARIMA-hi-90", "AutoARIMA-lo-80","AutoARIMA-hi-80"]
    assert forecast.shape == (n_series * horizon, len(expected))

    n_windows = 2
    cv = fa.as_pandas(sf.cross_validation(df=series, n_windows=n_windows, h=horizon, level=[80]))
    assert cv.shape[0] == n_series * n_windows * horizon
    assert cv.columns.tolist() == ['unique_id', 'ds', 'cutoff', 'y', 'AutoARIMA', 'AutoARIMA-lo-80', 'AutoARIMA-hi-80']

def pipeline_fitted(series, X_df, horizon):
    models = [SeasonalNaive(season_length=7)]
    pd_series = fa.as_pandas(series)
    pd_X = None if X_df is None else fa.as_pandas(X_df)
    sf = StatsForecast(models=models, freq='D')
    sf.forecast(df=pd_series, h=horizon, X_df=pd_X, level=[80, 90], fitted=True)
    fitted = sf.forecast_fitted_values()
    sf.forecast(df=series, h=horizon, X_df=X_df, level=[80, 90], fitted=True)
    distributed_fitted = (
        fa.as_pandas(sf.forecast_fitted_values())
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
        [fitted.columns]
        .astype(fitted.dtypes)  # fugue returns nullable and pyarrow dtypes
    )
    pd.testing.assert_frame_equal(fitted, distributed_fitted, atol=1e-5)
