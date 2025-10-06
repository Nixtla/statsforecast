import time
from functools import partial
from multiprocessing import cpu_count

import fire
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA as auto_arima_nixtla

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Other') -> None:

    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    train['ds'] = pd.to_datetime(train['ds']) 
    df_warmup=train.groupby("unique_id").head(10)

    models = [auto_arima_nixtla(season_length=seasonality)]
    start = time.time()
    fcst = StatsForecast(models=models, freq=freq, n_jobs=-1)
    # Warmup
    _ = fcst.forecast(df=df_warmup, h=1)
    forecasts = fcst.forecast(df=train, h=horizon)
    end = time.time()
    print(end - start)

    forecasts.columns = ['unique_id', 'ds', 'auto_arima_nixtla']
    forecasts.to_csv(f'data/statsforecast-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['auto_arima_nixtla']})
    time_df.to_csv(f'data/statsforecast-time-{dataset}-{group}.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)
