import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import fire
import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape
from prophet import Prophet
from sklearn.model_selection import ParameterGrid

from src.data import get_data

params_grid = {'seasonality_mode': ['multiplicative','additive'],
               'growth': ['linear', 'flat'],
               'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
               'n_changepoints': [5, 10, 15, 20]}
grid = ParameterGrid(params_grid)


def fit_and_predict(index, ts, horizon, freq, seasonality): 

    df = ts.drop('unique_id', 1)
    df['ds'] = pd.date_range(start='1970-01-01', periods=ts.shape[0], freq=freq)
    df_val = df.tail(horizon)
    df_train = df.drop(df_val.index)
    y_val = df_val['y'].values

    if len(df_train) >= horizon:
        val_results = {'losses': [], 'params': []}

        for params in grid:
            model = Prophet(seasonality_mode=params['seasonality_mode'],
                            growth=params['growth'],
                            weekly_seasonality=True,
			    daily_seasonality=True,
			    yearly_seasonality=True,
                            n_changepoints=params['n_changepoints'],
                            changepoint_prior_scale=params['changepoint_prior_scale'])
            model = model.fit(df_train)
            
            forecast = model.make_future_dataframe(periods=horizon, 
                                                   include_history=False, 
                                                   freq=freq)
            forecast = model.predict(forecast)
            forecast['unique_id'] = index
            forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])
            
            loss = mape(y_val, forecast['yhat'].values) 
            
            val_results['losses'].append(loss)
            val_results['params'].append(params)

        idx_params = np.argmin(val_results['losses']) 
        params = val_results['params'][idx_params]
    else:
        params = {'seasonality_mode': 'multiplicative',
                  'growth': 'flat',
                  'n_changepoints': 150,
                  'changepoint_prior_scale': 0.5}
    model = Prophet(seasonality_mode=params['seasonality_mode'],
                    growth=params['growth'],
		    weekly_seasonality=True,
		    daily_seasonality=True,
		    yearly_seasonality=True,
		    n_changepoints=params['n_changepoints'],
		    changepoint_prior_scale=params['changepoint_prior_scale'])
    model = model.fit(df)
    
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
