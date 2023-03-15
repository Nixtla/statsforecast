import ray

from statsforecast.utils import generate_series
from .utils import pipeline

def test_ray_flow():
    n_series = 2
    horizon = 7
    series = generate_series(n_series).reset_index()
    series['unique_id'] = series['unique_id'].astype(str)
    series = ray.data.from_pandas(series).repartition(2)
    pipeline(series, n_series, horizon)
