import time
from functools import partial
from multiprocessing import cpu_count

import fire
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsforecast import StatsForecast

from src.data import get_data


def auto_arima_pmdarima(X, h, future_xreg, season_length):
    y = X[:, 0] if X.ndim == 2 else X
    xreg = X[:, 1:] if (X.ndim == 2 and X.shape[1] > 1) else None
    mod = auto_arima(y, X=xreg,
                     method='bfgs',
                     information_criterion='aicc',
                     m=season_length, 
                     with_intercept=False, 
                     error_action='ignore')

    return mod.predict(h, X=future_xreg)

def main(group: str) -> None:
    train = get_data('data/', group)
    train['ds'] = pd.to_datetime(train['ds']) 
    train = train.set_index('unique_id')
    print(train.head())

    future_xreg = get_data('data/', group, False).drop(['y'], 1).set_index('unique_id')
    print(future_xreg.head())
    
    models = [
        (auto_arima_pmdarima, 24)
    ]

    start = time.time()
    fcst = StatsForecast(train, models=models, freq='H', n_jobs=1)
    forecasts = fcst.forecast(24 * 7, future_xreg)
    end = time.time()
    print(end - start)

    forecasts = forecasts.reset_index()
    forecasts.columns = ['unique_id', 'ds', 'auto_arima_pmdarima']
    forecasts.to_csv(f'data/pmdarima-forecasts-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['auto_arima_pmdarima']})
    time_df.to_csv(f'data/pmdarima-time-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
