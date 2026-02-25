
from itertools import product

import numpy as np
import pandas as pd
from datasetsforecast.losses import mape, smape

from src.data import get_data


def evaluate(lib: str, dataset: str, group: str):
    y_test, horizon, freq, seasonality = get_data('data/', dataset, group, False)
    y_test = y_test['y'].values.reshape(-1, horizon)

    if '-r' not in lib:
        try:
            forecast = pd.read_csv(f'data/{lib}-forecasts-{dataset}-{group}.csv')
            y_hat = forecast[lib].values.reshape(-1, horizon)
        except:
            return None
    else:
        try:
            y_hat = np.loadtxt(f'data/{lib}-forecasts-{dataset}-{group}.txt')
        except:
            return None
    

    evals = {}
    for metric in (mape, smape):
        metric_name = metric.__name__
        loss = metric(y_test, y_hat)#.mean()
        evals[metric_name] = loss 

    evals = pd.DataFrame(evals, index=[f'{dataset}_{group}']).rename_axis('dataset').reset_index()
    times = pd.read_csv(f'data/{lib}-time-{dataset}-{group}.csv')
    evals = pd.concat([evals, times], axis=1)

    return evals


if __name__ == '__main__':
    groups = ['Monthly', 'Yearly', 'Other', 'Quarterly']
    lib = ['Theta', 'OptimizedTheta', 'DynamicTheta', 'DynamicOptimizedTheta', 'ThetaEnsemble']
    datasets = ['M3']
    evaluation = [evaluate(lib, dataset, group) for lib, group in product(lib, groups) for dataset in datasets]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[['dataset', 'model', 'mape', 'smape', 'time']]
    evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(3)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation.to_csv('data/evaluation.csv')
    print(evaluation.query('metric=="smape"').T)
    print(evaluation.query('metric=="time"').T)
    print(evaluation)
