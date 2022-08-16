import os
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import fire
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet

from src.data import get_data


def convert_dates(index, df, horizon, freq, seasonality, dataset, group): 
    if dataset == 'M4' and group == 'Yearly':
        #yearly dataset has too long series
        #see https://eng.uber.com/m4-forecasting-competition/
        df = df.tail(60)
    df['ds'] = pd.date_range(end='2018-01-01', periods=df.shape[0], freq=freq)
    return df

def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    if dataset == 'M4':
        #add date to this dataset
        partial_convert_dates = partial(convert_dates, horizon=horizon, freq=freq,
                                        seasonality=seasonality, 
                                        dataset=dataset, group=group)
        with Pool(cpu_count()) as pool:
            train = pool.starmap(partial_convert_dates, train.groupby('unique_id'))
            train = pd.concat(train)
    else:
        train['ds'] = pd.to_datetime(train['ds'])
    train = train.rename(columns={'unique_id': 'ID'})

    start = time.time()
    if dataset == 'ERCOT':
        m = NeuralProphet(
            n_forecasts=24,
            n_lags=7*24,
            learning_rate=0.01,
            num_hidden_layers=1,
            d_hidden=16,
	)
        regions = list(train)[1:-2]
        m = m.add_lagged_regressor(names=regions)#, only_last_value=True)
        m = m.highlight_nth_step_ahead_of_each_forecast(24)
    else:
        m = NeuralProphet(n_lags=max(horizon, seasonality), n_forecasts=horizon)
    metrics = m.fit(train, freq=freq)
    future = m.make_future_dataframe(df=train, periods=horizon)
    forecasts = m.predict(df=future, decompose=False)
    end = time.time()
    print(end - start)

    forecasts = forecasts.groupby('ID').tail(horizon)
    forecasts['yhat'] = forecasts.filter(regex='yhat*').max(axis=1)
    forecasts = forecasts.filter(items=['ID', 'ds', 'yhat'])
    forecasts.columns = ['unique_id', 'ds', 'neuralprophet']
    forecasts.to_csv(f'data/neuralprophet-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['neuralprophet']})
    time_df.to_csv(f'data/neuralprophet-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    fire.Fire(main)
