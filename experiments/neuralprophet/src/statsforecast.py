import os
import time
os.environ['NUMBA_RELEASE_GIL'] = 'True'
os.environ['NUMBA_CACHE'] = 'True'

import fire
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ets as ets_statsforecast
from statsforecast.utils import AirPassengers as ap

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')
    if dataset == 'Tourism':
        #we add a constant to avoid zero errors
        constant = 100
        train['y'] += constant
    
    models = [
        (ets_statsforecast, seasonality)
    ]
    ets_statsforecast(ap.astype(np.float32), 12, season_length=seasonality)

    start = time.time()
    fcst = StatsForecast(train, models=models, freq=freq, n_jobs=os.cpu_count())
    forecasts = fcst.forecast(horizon)
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts.columns = ['unique_id', 'ds', 'ets_statsforecast']
    if dataset == 'Tourism':
        #remove constant from forecasts
        forecasts['ets_statsforecast'] -= constant
    forecasts.to_csv(f'data/statsforecast-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['ets_statsforecast']})
    time_df.to_csv(f'data/statsforecast-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
