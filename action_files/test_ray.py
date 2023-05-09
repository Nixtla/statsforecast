import ray

from statsforecast.utils import generate_series
from .utils import pipeline

def test_ray_flow():
    n_series = 2
    horizon = 7
    series = generate_series(n_series).reset_index()
    series['unique_id'] = series['unique_id'].astype(str)
    ctx = ray.data.context.DatasetContext.get_current()
    ctx.use_streaming_executor = False
    series = ray.data.from_pandas(series).repartition(2)
    pipeline(series, n_series, horizon)
