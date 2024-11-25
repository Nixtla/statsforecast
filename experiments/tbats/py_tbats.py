import os
import time
from multiprocessing import cpu_count
import concurrent.futures
os.environ['NIXTLA_NUMBA_RELEASE_GIL'] = '1'
os.environ['NIXTLA_NUMBA_CACHE'] = '1'

import fire
import pandas as pd
from tbats import TBATS as TBATSPY
from data import get_data

def generate_fcst(vals):
    unique_id, train, horizon, seasonal_periods = vals
    train_subset = train[train['unique_id'] == unique_id]
    estimator = TBATSPY(seasonal_periods=[seasonal_periods])
    fitted_model = estimator.fit(train_subset['y'].values)
    fcst = fitted_model.forecast(steps=horizon)
    return pd.DataFrame({
        'unique_id': [unique_id] * len(fcst),
        'y': fcst
    })

def main(dataset: str = 'M3', group: str = 'Other') -> None:

    seasonality = {
        'Yearly': 1, 
        'Quarterly': 4,
        'Monthly': 12,
        'Other': 1
        }
    
    train, horizon, _, _ = get_data('data/', dataset, group)
    seasonal_periods = seasonality[group]
    vals_list = [(uid, train, horizon, seasonal_periods) for uid in train['unique_id'].unique()]

    forecasts = pd.DataFrame()
    start = time.time()  
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        results = executor.map(generate_fcst, vals_list)
        for result in results:
            forecasts = pd.concat([forecasts, result], ignore_index=True)
    end = time.time() 

    forecasts.to_csv(f'data/PY-TBATS-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': 'PY-TBATS'})
    time_df.to_csv(f'data/PY-TBATS-time-{dataset}-{group}.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)
