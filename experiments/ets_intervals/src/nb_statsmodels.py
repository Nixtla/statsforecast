import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import fire
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.ets import AutoETS
from statsforecast import StatsForecast
from statsforecast.models import Naive

from src.data import get_data

from tqdm import tqdm 

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
    levels = np.arange(55, 100, 5)
    alpha = np.flip(np.arange(0.05, 0.5, 0.05))
    x = ts['y'].values
    try:
        x = pd.Series(x)  
        mod = auto_ets(x, seasonality)  
        forecast = mod.get_prediction(start=x.index[-1]+1, end=x.index[-1]+horizon) 
        res = {}

        for k in range(0, len(levels)): 
            pred_int = forecast.summary_frame(alpha = alpha[k]) 
            pred_int = pred_int.reset_index()
    
            res['ds'] = (pred_int['index'].values)+1
            res['statsmodels_mean'] = pred_int['mean'].values
            res['statsmodels_lowerb_'+str(levels[k])] = pred_int['pi_lower'].values
            res['statsmodels_upperb_'+str(levels[k])] = pred_int['pi_upper'].values
    
        df = pd.DataFrame.from_dict(res)
        df['unique_id'] = np.repeat(ts['unique_id'].unique(), horizon)
        df = df[ ['unique_id', 'ds', 'statsmodels_mean']+['statsmodels_lowerb_'+str(i) for i in levels]+['statsmodels_upperb_'+str(i) for i in levels]]   
    except:
        x = ts
        x['ds'] = x['ds'].astype(int)
        models = [Naive()]
        fcst = StatsForecast(df=x, models=models, freq=freq, n_jobs=cpu_count())
        df = fcst.forecast(h=horizon, level=levels)
        df = df.reset_index()
        df.columns = ['unique_id', 'ds', 'statsmodels_mean']+['statsmodels_lowerb_'+str(i) for i in levels]+['statsmodels_upperb_'+str(i) for i in levels]
        df = df[['unique_id', 'ds', 'statsmodels_mean']+['statsmodels_lowerb_'+str(i) for i in levels]+['statsmodels_upperb_'+str(i) for i in levels]]

    return df 

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
    forecasts.to_csv(f'data/statsmodels-forecasts-{dataset}-{group}-pred-int.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['ets_statsmodels']})
    time_df.to_csv(f'data/statsmodels-time-{dataset}-{group}-pred-int.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)