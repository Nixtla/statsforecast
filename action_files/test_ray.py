from os import cpu_count

import pytest
import ray
from statsforecast.core import StatsForecast, _get_n_jobs
from statsforecast.models import ( 
    ADIDA,
    AutoARIMA,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    ETS,
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
from statsforecast.utils import generate_series

@pytest.mark.parametrize(
	'test_input, expected', 
	[
		("(10, -1, 'auto')", cpu_count()), 
		("(10, None, 'auto')", cpu_count()),
		("(1, -1, 'auto')", 1),
		("(1, None, 'auto')", 1),
		("(2, 10, 'auto')", 2),
	]
)
def test_ray_n_jobs(test_input, expected):
	ray.init(ignore_reinit_error=True)
	assert _get_n_jobs(*eval(test_input)) == expected
	ray.shutdown()

def test_ray_flow():
    n_series = 20
    horizon = 7
    series = generate_series(20)
    models = [
		ADIDA(), AutoARIMA(season_length=7), 
		CrostonClassic(), CrostonOptimized(),
		CrostonSBA(), ETS(season_length=7),
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
    ray_context = ray.init(ignore_reinit_error=True)
    fcst = StatsForecast(
        df=series,
        models=models,
        freq='D',
        n_jobs=-1,
        ray_address=ray_context.address_info['address']
    )
    forecast = fcst.forecast(7)
    ray.shutdown()
    assert forecast.shape == (n_series * horizon, len(models) + 1)

