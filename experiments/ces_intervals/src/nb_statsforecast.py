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
from statsforecast.models import AutoCES, Naive
from statsforecast.utils import AirPassengers as ap

from src.data import get_data
from datasetsforecast.m4 import M4

def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')
    
    models = [AutoCES(season_length=seasonality)]
    levels = [k for k in range(55,100,5)]
    start = time.time()
    fcst = StatsForecast(df=train, models=models, freq=freq, n_jobs=cpu_count(), fallback_model = Naive())
    forecasts = fcst.forecast(h=horizon, level=levels)
    end = time.time()

    forecasts = forecasts.reset_index()
    forecasts.columns = ['unique_id', 'ds', 'statsforecast_mean']+['statsforecast_lowerb_'+str(i) for i in reversed(levels)]+['statsforecast_upperb_'+str(i) for i in levels]
    forecasts = forecasts[['unique_id', 'ds', 'statsforecast_mean']+['statsforecast_lowerb_'+str(i) for i in levels]+['statsforecast_upperb_'+str(i) for i in levels]]
    forecasts.to_csv(f'data/statsforecast-ces-forecasts-{dataset}-{group}-pred-int.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['ces_statsforecast']})
    time_df.to_csv(f'data/statsforecast-ces-time-{dataset}-{group}-pred-int.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)