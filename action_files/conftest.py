import os

import numpy as np
import pandas as pd
import pytest

from statsforecast.utils import generate_series

os.environ['NIXTLA_ID_AS_COL'] = '1'


@pytest.fixture
def n_series():
    return 2

@pytest.fixture
def horizon():
    return 7

@pytest.fixture()
def local_data(n_series, horizon):
    n_static = 2
    series = generate_series(n_series, n_static_features=n_static)
    static_features = []
    for i in range(n_static):
        name = f'static_{i}'
        series[name] = series[name].astype(int)
        static_features.append(name)
    series['unique_id'] = series['unique_id'].astype(str)
    uids = series['unique_id'].unique()
    static_values = series.groupby('unique_id')[static_features].head(1)
    static_values['unique_id'] = uids
    last_train_dates = series.groupby('unique_id')['ds'].max()
    pred_start = last_train_dates + pd.offsets.Day()
    pred_end = last_train_dates + horizon * pd.offsets.Day()
    pred_dates = np.hstack([pd.date_range(start, end) for start, end in zip(pred_start, pred_end)])
    X_df = pd.DataFrame(
        {
            'unique_id': np.repeat(uids, horizon),
            'ds': pred_dates,
        }
    )
    X_df = X_df.merge(static_values, on='unique_id')
    return series, X_df
