import fugue.api as fa
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


def pipeline(series, n_series, horizon):
    models = [
		ADIDA(), AutoARIMA(season_length=7), 
                ARIMA(season_length=7, order=(0, 1, 2)),
		CrostonClassic(), CrostonOptimized(),
		CrostonSBA(), AutoETS(season_length=7),
		HistoricAverage(), 
		IMAPA(), Naive(), 
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
    forecast = fa.as_pandas(sf.forecast(df=series, h=horizon, level=[80, 90]))
    print(forecast)
    assert forecast.shape == (n_series * horizon, len(models) + 2)

