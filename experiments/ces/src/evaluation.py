from itertools import product

import numpy as np
import pandas as pd
from neuralforecast.data.datasets.m4 import M4Evaluation
from neuralforecast.losses.numpy import rmse, mape, smape

from src.data import get_data


def evaluate(model: str, dataset: str, group: str):
    y_test, horizon, freq, seasonality = get_data('data/', dataset, group, False)
    if model != 'ces-r':
        try:
            forecast = pd.read_csv(f'data/{model}-forecasts-{dataset}-{group}.csv')
        except:
            return None
        col = model
        y_hat = forecast[col].values.reshape(-1, horizon)
    else:
        try:
            y_hat = np.loadtxt(f'data/{model}-forecasts-M4-{group}.txt')
        except:
            return None

    y_test = y_test['y'].values.reshape(-1, horizon)
    assert np.isnan(y_hat).sum() == 0, f'{model},{dataset},{group}'

    evals = M4Evaluation.evaluate('data', group, y_hat)
    evals = evals.rename_axis('dataset').reset_index()
    times = pd.read_csv(f'data/{model}-time-{dataset}-{group}.csv')
    evals = pd.concat([evals, times], axis=1)
    evals['model'] = model

    return evals


if __name__ == '__main__':
    groups = ['Yearly', 'Quarterly', 'Monthly', 'Daily', 'Weekly', 'Hourly']
    models = ['ensemble', 'ets', 'ces', 'ces-r']
    datasets = ['M4']
    evaluation = [evaluate(model, dataset, group) for model, group in product(models, groups) for dataset in datasets]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[['dataset', 'model', 'MASE', 'time']]
    evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(3)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation = evaluation[['dataset', 'metric'] + models]
    evaluation.to_csv('data/evaluation.csv')
    print(evaluation.to_markdown(index=False))
