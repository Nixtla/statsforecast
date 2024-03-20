import os
import time
from multiprocessing import cpu_count
os.environ['NIXTLA_NUMBA_RELEASE_GIL'] = '1'
os.environ['NIXTLA_NUMBA_CACHE'] = '1'
os.environ['NIXTLA_ID_AS_COL'] = '1'

import fire
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoTBATS

from data import get_data

def find_seasonality(group: str) -> int:
    seasonality = {
        'Yearly': [1], 
        'Quarterly': [4],
        'Monthly': [12],
        'Weekly': [1],
        'Daily': [7],
        'Hourly': [24, 24*7], 
        'Other': [1]
    }

    if group in seasonality:
        return seasonality[group]
    else:
        raise ValueError(f'Group {group} not found')

def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, frequency, _ = get_data('data/', dataset, group)

    if dataset == 'M4':
        train['ds'] = train['ds'].astype(int)
        frequency = 1 # values in ds column are integers

    seasonality = find_seasonality(group)
    models = [AutoTBATS(season_length=seasonality)]
    
    start = time.time()
    fcst = StatsForecast(models=models, freq=frequency, n_jobs=cpu_count()) 
    forecasts = fcst.forecast(df=train, h=horizon)
    end = time.time()

    forecasts.to_csv(f'data/AutoTBATS-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['AutoTBATS']})
    time_df.to_csv(f'data/AutoTBATS-time-{dataset}-{group}.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)