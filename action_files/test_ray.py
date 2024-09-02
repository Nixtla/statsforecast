import sys

import pytest

if sys.version_info >= (3, 12):
    pytest.skip("Ray requires Python<3.12", allow_module_level=True)
import ray

from .utils import pipeline, pipeline_with_level, pipeline_fitted


def to_distributed(df):
    return ray.data.from_pandas(df).repartition(2)


@pytest.fixture()
def sample_data(local_data):
    series, X_df = local_data
    return to_distributed(series), to_distributed(X_df)


def test_ray_flow(horizon, sample_data, n_series):
    pipeline(*sample_data, n_series, horizon)


def test_ray_flow_with_level(horizon, sample_data, n_series):
    pipeline_with_level(*sample_data, n_series, horizon)


@pytest.mark.parametrize("use_x", [True, False])
def test_ray_flow_with_fitted(horizon, use_x, sample_data):
    series, X_df = sample_data
    if not use_x:
        X_df = None
    pipeline_fitted(series, X_df, horizon)
