import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import fire
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

from src.data import get_data


def fit_and_predict(index, ts, horizon, freq, seasonality): 
    x = ts['y'].values
    mod = auto_arima(x, m=seasonality, with_intercept=False, error_action='ignore')
    forecast = mod.predict(horizon)
    ds = pd.date_range(start=ts.ds.max(), periods=horizon + 1, freq=freq)[1:]
    df = pd.DataFrame({
            "unique_id": index,
            "ds": ds,
            "auto_arima_pmdarima": forecast,
    })
    return df

def main(dataset: str, group: str) -> None:
    train, horizon, freq, seasonality = get_data('data', dataset, group)
    partial_fit_and_predict = partial(
        fit_and_predict,
        horizon=horizon,
        freq=freq,
        seasonality=seasonality
    )
    start = time.time()
    print(f'Parallelism on {cpu_count()} CPU')
    with Pool(cpu_count()) as pool:
        results = pool.starmap(partial_fit_and_predict, train.groupby('unique_id'))
    end = time.time()
    
    forecasts = pd.concat(results)
    forecasts.to_csv(f'data/pmdarima-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['auto_arima_pmdarima']})
    time_df.to_csv(f'data/pmdarima-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
