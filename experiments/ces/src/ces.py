import os
import time
from functools import partial
from multiprocessing import cpu_count
os.environ['NUMBA_RELEASE_GIL'] = 'True'
#os.environ['NUMBA_CACHE'] = 'True'

import fire
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoCES
from statsforecast.utils import AirPassengers as ap

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    print(group, seasonality)
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')
    
    models = [AutoCES(season_length=seasonality)]
    start = time.time()
    fcst = StatsForecast(df=train, models=models, freq=freq, n_jobs=cpu_count())
    forecasts = fcst.forecast(h=horizon)
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts.columns = ['unique_id', 'ds', 'ces']
    forecasts.to_csv(f'data/ces-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['ces']})
    time_df.to_csv(f'data/ces-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    AutoCES(season_length=12).forecast(ap.astype(np.float32), h=12)
    fire.Fire(main)
