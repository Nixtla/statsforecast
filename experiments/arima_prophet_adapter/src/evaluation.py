from itertools import product

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape, rmse, smape, mae, mase
from neuralforecast.data.datasets.m4 import M4Evaluation, M4Info

from src.data import get_data, dict_datasets


def evaluate(lib: str, dataset: str, group: str):
    try:
        forecast = pd.read_csv(f'data/{lib}-forecasts-{dataset}-{group}.csv')
    except:
        return None

    y_train, horizon, freq, seasonality = get_data('data/', dataset, group)
    y_train = y_train['y'].values
    y_test, *_ = get_data('data/', dataset, group, False)
    y_test = y_test['y'].values.reshape(-1, horizon)
    y_hat = forecast[lib].values.reshape(-1, horizon)

    evals = {}
    for metric in (mape, smape):
        metric_name = metric.__name__
        if metric_name == 'mase':
            loss = mase(y_test, y_hat, y_train, seasonality=seasonality)
            loss = loss.mean()
        else:
            loss = metric(y_test, y_hat, axis=1).mean()
        evals[metric_name] = loss 
    evals = pd.DataFrame(evals, index=[f'{dataset}_{group}']).rename_axis('dataset').reset_index()
    
    times = pd.read_csv(f'data/{lib}-time-{dataset}-{group}.csv')
    evals = pd.concat([evals, times], axis=1)

    return evals


if __name__ == '__main__':
    groups = ['Yearly', 'Quarterly', 'Monthly', 'Other', 'Daily', 'Hourly', 'Weekly']
    lib = ['arima_prophet_adapter', 'prophet']
    datasets = ['M3', 'Tourism', 'M4', 'PeytonManning']
    evaluation = [evaluate(lib, dataset, group) for lib, group in product(lib, groups) for dataset in datasets]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[['dataset', 'model', 'mape', 'smape', 'time']]
    evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(2)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation.to_csv('data/evaluation.csv')
