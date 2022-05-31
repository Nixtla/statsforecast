import fire
import numpy as np
import pandas as pd
from pathlib import Path

from statsforecast.core import StatsForecast
from statsforecast.models import seasonal_naive, linear_auto_regression, seasonal_window_average, auto_arima, naive

from .utils import forecast_r, DATASETS


def linear_regression(X, h, future_xreg):
    y = X[:, 0]
    xreg = X[:, 1:]
    coefs, *_ = np.linalg.lstsq(xreg, y, rcond=None)
    return future_xreg @ coefs

def main(dataset: str, model: str):
    df = pd.read_csv(f'./data/{dataset}/M/df_y.csv')
    #df_x = pd.read_csv(f'./data/{dataset}/M/df_x.csv')
    #df = df.merge(df_x, how='left', on=['ds'])
    df = df.sort_values(['unique_id', 'ds']).set_index('unique_id')
    #df['ds'] = pd.to_datetime(df['ds'])
    #df['intercept'] = 1
    #df['week'] = df['ds'].dt.weekofyear
    #df = pd.get_dummies(df, columns=['week'], drop_first=True)
    meta = DATASETS[dataset]
    freq = meta['freq']
    seasonality = meta['seasonality']
    horizons = meta['horizons']
    test_size = meta['test_size']
    fcst = StatsForecast(df, [(forecast_r, seasonality, model)], freq=freq, n_jobs=-1)
    results = {}
    dir_ = Path(f'./results/{dataset}/{model}')
    dir_.mkdir(exist_ok=True, parents=True)
    for horizon in horizons:
        cv = fcst.cross_validation(h=horizon, test_size=test_size)
        cv.reset_index().to_csv(dir_ / f'./cv-{horizon}.csv', index=False)
        models = cv.drop(['ds', 'cutoff', 'y'], axis=1).columns
        for model_name in models:
            error = cv['y'] - cv[model_name]
            mae = np.mean(np.abs(error))
            mse = np.mean(error**2)
            results[horizon] = {'mse': mse, 'mae': mae}

    results = pd.DataFrame(results).T
    results.index.name = 'horizon'
    results = results.reset_index()
    results.insert(0, 'dataset', dataset)
    results.insert(0, 'model', model)
    print(results)
    return results

if __name__=="__main__":
    for model in ['snaive', 'ses', 'holt', 'hw', 'splinef', 'nnetar', 'ets', 'stlm', 'thetaf', 'auto.arima', 'tbats']:
        results = []
        for dataset in DATASETS.keys():
            print(f'model: {model}, dataset: {dataset}')
            results.append(main(dataset, model))
        results = pd.concat(results)
        print(results)
        results.to_csv(f'./results/performance-{model}.csv', index=False)


