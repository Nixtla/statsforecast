import pytest
import ray

from statsforecast.utils import generate_series
from .utils import pipeline, pipeline_with_level

@pytest.fixture()
def n_series():
    return 2

@pytest.fixture()
def sample_data(n_series):
    series = generate_series(n_series).reset_index()
    series['unique_id'] = series['unique_id'].astype(str)
    series = ray.data.from_pandas(series).repartition(2)
    return series

def test_ray_flow(sample_data, n_series):
    horizon = 7
    pipeline(sample_data, n_series, horizon)

def test_ray_flow_with_level(sample_data, n_series):
    horizon = 7
    pipeline_with_level(sample_data, n_series, horizon)