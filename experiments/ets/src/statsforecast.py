import os
import time
from functools import partial
from multiprocessing import cpu_count
os.environ['NUMBA_RELEASE_GIL'] = 'True'
os.environ['NUMBA_CACHE'] = 'True'

import fire
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ets as ets_nixtla
from statsforecast.utils import AirPassengers as ap

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')
    
    models = [
        (ets_nixtla, seasonality)
    ]
    ets_nixtla(ap.astype(np.float32), 12, season_length=seasonality)

    start = time.time()
    fcst = StatsForecast(train, models=models, freq=freq, n_jobs=cpu_count())
    fcst.last_dates = pd.DatetimeIndex(fcst.last_dates)
    forecasts = fcst.forecast(horizon)
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts.columns = ['unique_id', 'ds', 'ets_nixtla']
    forecasts.to_csv(f'data/statsforecast-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['ets_nixtla']})
    time_df.to_csv(f'data/statsforecast-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
