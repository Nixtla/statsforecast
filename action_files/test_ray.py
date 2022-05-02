from os import cpu_count

import pytest
import ray
from statsforecast.core import StatsForecast, _get_n_jobs
from statsforecast.models import (
    adida,
    auto_arima,
    croston_classic,
    croston_optimized,
    croston_sba,
    historic_average,
    imapa,
    naive,
    random_walk_with_drift,
    seasonal_exponential_smoothing,
    seasonal_naive,
    seasonal_window_average,
    ses,
    tsb,
    window_average,
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
    models = [
        adida, croston_classic, croston_optimized,
        croston_sba, historic_average, imapa, naive, 
        random_walk_with_drift, (seasonal_exponential_smoothing, 7, 0.1),
        (seasonal_naive, 7), (seasonal_window_average, 7, 4),
        (ses, 0.1), (tsb, 0.1, 0.3), (window_average, 4)
    ]
    series = generate_series(20)
    ray_context = ray.init(ignore_reinit_error=True)
    fcst = StatsForecast(
        series,
        models=models,
        freq='D',
        n_jobs=-1,
        ray_address=ray_context.address_info['address']
    )
    forecast = fcst.forecast(7)
    ray.shutdown()
    assert forecast.shape == (n_series * horizon, len(models) + 1)

