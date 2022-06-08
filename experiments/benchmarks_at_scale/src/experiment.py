import os
from pathlib import Path
from time import time

import pandas as pd
from statsforecast.core import StatsForecast
from statsforecast.models import (
        croston_classic, seasonal_naive, naive,
        adida, historic_average,
        seasonal_window_average,
        seasonal_exponential_smoothing,
        imapa,
        window_average,
        random_walk_with_drift,
)


if __name__=="__main__":
    dir_ = Path('./results')
    dir_.mkdir(exist_ok=True)
    for length in [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]:
        print(f'length: {length}')
        file_ = dir_ / f'forecasts_{length}.parquet'
        if file_.exists():
            print('Already generated')
            continue
        series = pd.read_parquet(f'./data/series_{length}.parquet')
        #series['y'] += 0.0001 #prevent division by zero error
        print('Data read')
        models = [
            croston_classic,
            (seasonal_naive, 7),
            naive, adida,
            historic_average,
            (seasonal_window_average, 7, 4),
            imapa,
            (window_average, 7),
            (seasonal_exponential_smoothing, 7, 0.9),
            #random_walk_with_drift
        ]
        init = time()
        model = StatsForecast(series,
                              models=models,
                              freq='D',
                              n_jobs=-1) 
        end = time()
        print(f'Init forecast, instantiation time: {end-init}')
        init = time()
        forecasts = model.cross_validation(h=7, test_size=13)
        total_time = (time() - init) / 60
        time_df = pd.DataFrame({'time': [total_time], 'length': [length], 'cpus': [os.cpu_count()]})
        time_df.to_csv(f'./results/time_{length}.csv')
        forecasts.to_parquet(f'./results/forecasts_{length}.parquet')
        print(f'n_series: {length} total time: {total_time}')
