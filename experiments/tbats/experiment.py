import os
import time
from multiprocessing import cpu_count
os.environ['NIXTLA_NUMBA_RELEASE_GIL'] = '1'
os.environ['NIXTLA_NUMBA_CACHE'] = '1'

import fire
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoTBATS, SeasonalNaive
from data import get_data

def find_seasonality(model, group) -> int:
    seasonality = {
        'Yearly': 1, 
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 7,
        'Hourly': [24, 24*7], 
        'Other': 1
    }

    if model == 'SeasonalNaive': 
        seasonality['Hourly'] = 24

    if group in seasonality:
        return seasonality[group]
    else:
        raise ValueError(f'Group {group} not found')


def main(dataset: str = 'M3', group: str = 'Other', model: str='AutoTBATS') -> None:
    train, horizon, frequency, _ = get_data('data/', dataset, group)

    if dataset == 'M4':
        train['ds'] = train['ds'].astype(int)
        frequency = 1 # since values in ds column are integers

    seasonality = find_seasonality(model, group)
    if model == 'AutoTBATS': 
        models = [AutoTBATS(season_length=seasonality)]
    elif model == 'SeasonalNaive': 
        models = [SeasonalNaive(season_length=seasonality)]
    else: 
        raise ValueError(f'This experiment only supports AutoTBATS and SeasonalNaive')
    
    start = time.time()
    fcst = StatsForecast(models=models, freq=frequency, n_jobs=cpu_count()) 
    forecasts = fcst.forecast(df=train, h=horizon)
    end = time.time()

    forecasts.to_csv(f'data/{model}-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': [model]})
    time_df.to_csv(f'data/{model}-time-{dataset}-{group}.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)
