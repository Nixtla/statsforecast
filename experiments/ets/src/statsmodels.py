import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import fire
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.ets import AutoETS
from statsforecast.models import random_walk_with_drift

from src.data import get_data

def auto_ets(y, m):
    data_positive = min(y) > 0
    errortype = ['add', 'mul']
    trendtype = [None, 'add']
    seasontype = [None, 'add', 'mul']
    damped = [True, False]
    best_ic = np.inf
    for etype in errortype:
        for ttype in trendtype:
            for stype in seasontype:
                for dtype in damped:
                    if ttype is None and dtype:
                        continue
                    if etype == 'add' and (ttype == 'mul' and stype == 'mul'):
                        continue
                    if etype == 'mul' and ttype == 'mul' and stype == 'add':
                        continue
                    if (not data_positive) and etype == 'mul':
                        continue
                    if stype in ['add', 'mul'] and m == 1:
                        continue
                    model = ETSModel(y, error=etype, trend=ttype, damped_trend=dtype, 
                                    seasonal=stype, seasonal_periods=m)
                    model = model.fit(disp=False)
                    fit_ic = model.aicc
                    if not np.isnan(fit_ic):
                        if fit_ic < best_ic:
                            best_model = model
    return best_model

def fit_and_predict(index, ts, horizon, freq, seasonality): 
    x = ts['y'].values
    try:
        mod = auto_ets(x, seasonality)
        forecast = mod.forecast(horizon)
    #print(forecast.flatten().flatten())
    except:
        forecast = random_walk_with_drift(x, horizon, None)

    forecast = pd.DataFrame({
        'ds': np.arange(ts['ds'].max() + 1, ts['ds'].max() + horizon + 1),
        'ypred': forecast
    })
    forecast['unique_id'] = index
    
    return forecast[['unique_id', 'ds', 'ypred']]

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
    forecasts.columns = ['unique_id', 'ds', 'ets_statsmodels']
    forecasts.to_csv(f'data/statsmodels-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['ets_statsmodels']})
    time_df.to_csv(f'data/statsmodels-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
