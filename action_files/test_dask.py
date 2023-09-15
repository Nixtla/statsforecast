import dask.dataframe as dd
import pytest

from .utils import pipeline, pipeline_with_level


def to_distributed(df):
    return dd.from_pandas(df, npartitions=2)

@pytest.fixture()
def sample_data(local_data):
    series, X_df = local_data
    return to_distributed(series), to_distributed(X_df)

def test_dask_flow(horizon, sample_data, n_series):
    pipeline(*sample_data, n_series, horizon)

def test_dask_flow_with_level(horizon, sample_data, n_series):
    pipeline_with_level(*sample_data, n_series, horizon)
