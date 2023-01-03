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


@pytest.fixture(scope="module")
def ray_fix():
    ray_context = ray.init(num_cpus=2, include_dashboard=False)
    yield ray_context
    ray.shutdown()

@pytest.mark.parametrize(
'test_input, expected', 
	[
		("(10, -1, 'auto')", 2), 
		("(10, None, 'auto')", 2),
		("(1, -1, 'auto')", 1),
		("(1, None, 'auto')", 1),
		("(2, 10, 'auto')", 2),
	]
)
def test_ray_n_jobs(test_input, expected, ray_fix):
	assert ray.is_initialized()
	assert _get_n_jobs(*eval(test_input)) == expected

def test_ray_flow(ray_fix):
    assert ray.is_initialized()
    n_series = 20
    horizon = 7
    series = generate_series(20)
    series['ds'] = series['ds'].astype(str)
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
    fcst = StatsForecast(
        df=series,
        models=models,
        freq='D',
        n_jobs=-1,
        ray_address=ray_fix.address_info['address']
    )
    forecast = fcst.forecast(7)
    assert forecast.shape == (n_series * horizon, len(models) + 1)

