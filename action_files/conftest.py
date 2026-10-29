import numpy as np
import pandas as pd
import pytest

from statsforecast.utils import generate_series


@pytest.fixture
def n_series():
    return 2

@pytest.fixture
def horizon():
    return 7

@pytest.fixture()
def local_data(n_series, horizon):
    rng = np.random.default_rng(0)
    series = generate_series(n_series)
    series['unique_id'] = series['unique_id'].astype(str)
    series['x'] = rng.standard_normal(len(series))
    uids = series['unique_id'].unique()
    last_train_dates = series.groupby('unique_id')['ds'].max()
    pred_start = last_train_dates + pd.offsets.Day()
    pred_end = last_train_dates + horizon * pd.offsets.Day()
    pred_dates = np.hstack([pd.date_range(start, end) for start, end in zip(pred_start, pred_end)])
    X_df = pd.DataFrame(
        {
            'unique_id': np.repeat(uids, horizon),
            'ds': pred_dates,
            'x': rng.standard_normal(n_series * horizon),
        }
    )
    return series, X_df
