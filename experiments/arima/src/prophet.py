import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import fire
import numpy as np
import pandas as pd
from prophet import Prophet

from src.data import get_data


def fit_and_predict(index, ts, horizon, freq, seasonality): 

    ts['ds'] = pd.date_range(start='1970-01-01', periods=ts.shape[0], freq=freq)
   
    model = Prophet()
    model.add_seasonality(name='season', 
                          period=seasonality,
                          fourier_order=5)
    model = model.fit(ts)
    
    forecast = model.make_future_dataframe(periods=horizon, 
                                           include_history=False, 
                                           freq=freq)
    forecast = model.predict(forecast)
    forecast['unique_id'] = index
    forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])

    return forecast

def main(dataset: str, group: str) -> None:
    train, horizon, freq, seasonality = get_data('data', dataset, group)
    
    partial_fit_and_predict = partial(fit_and_predict, 
                                      horizon=horizon, freq=freq, seasonality=seasonality)
    start = time.time()
    print(f'Parallelism on {cpu_count()} CPU')
    with Pool(cpu_count()) as pool:
        results = pool.starmap(partial_fit_and_predict, train.groupby('unique_id'))
    end = time.time()
    print(end - start)
    
    forecasts = pd.concat(results)
    forecasts.columns = ['unique_id', 'ds', 'prophet']
    forecasts.to_csv(f'data/prophet-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['prophet']})
    time_df.to_csv(f'data/prophet-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
   fire.Fire(main)
