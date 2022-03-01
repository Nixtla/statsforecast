import time
from functools import partial
from multiprocessing import cpu_count

import fire
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import auto_arima as auto_arima_nixtla

from src.data import get_data


def main(group: str) -> None:
    train = get_data('data/', group)#[['unique_id', 'ds', 'y']]
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')

    future_xreg = get_data('data/', group, False).drop(['y'], 1).set_index('unique_id')
    
    models = [
        (auto_arima_nixtla, 24)
    ]

    start = time.time()
    fcst = StatsForecast(train, models=models, freq='H', n_jobs=1)
    forecasts = fcst.forecast(24 * 7, future_xreg)
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts.columns = ['unique_id', 'ds', 'auto_arima_nixtla']
    forecasts.to_csv(f'data/statsforecast-forecasts-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['auto_arima_nixtla']})
    time_df.to_csv(f'data/statsforecast-time-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
