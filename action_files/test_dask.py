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
    renamer = {'unique_id': 'id', 'ds': 'time', 'y': 'value'}
    series, X_df = sample_data
    series = series.rename(columns=renamer)
    X_df = X_df.rename(columns=renamer)
    pipeline(series, X_df, n_series, horizon, id_col='id', time_col='time', target_col='value')

def test_dask_flow_with_level(horizon, sample_data, n_series):
    pipeline_with_level(*sample_data, n_series, horizon)
