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
    return forecast, index, ts

def main(dataset: str, group: str) -> None:
    train, horizon, freq, seasonality = get_data('data', dataset, group)
    print(train)
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
    print(end - start)

    ds = pd.date_range(start=train.ds.max(), periods=horizon + 1, freq=freq)[1:]
    preds = []
    for pred in results:
        preds.append(
            pd.DataFrame({
            "unique_id": pred[1],
            "ds": ds,
            "auto_arima_pmdarima": pred[0],
        }))
    
    forecasts = pd.concat(preds)
    forecasts.to_csv(f'data/pmdarima-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['auto_arima_pmdarima']})
    time_df.to_csv(f'data/pmdarima-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
