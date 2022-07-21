from itertools import product

import numpy as np
import pandas as pd
from neuralforecast.data.datasets.m4 import M4Evaluation, M4Info

from src.data import get_data


def evaluate(lib: str, group: str):
    if lib != 'ets-r':
        try:
            forecast = pd.read_csv(f'data/{lib}-forecasts-M4-{group}.csv')
        except:
            return None
        horizon = M4Info[group].horizon
        if lib == 'statsforecast':
            col = 'ets_nixtla'
        elif lib == 'prophet':
            col = 'prophet'
        else:
            col = 'ets_statsmodels'
        forecast = forecast[col].values.reshape(-1, horizon)
    else:
        try:
            forecast = np.loadtxt(f'data/{lib}-forecasts-M4-{group}.txt')
        except:
            return None
    evals = M4Evaluation.evaluate('data', group, forecast)
    times = pd.read_csv(f'data/{lib}-time-M4-{group}.csv')
    evals = evals.rename_axis('dataset').reset_index()
    evals = pd.concat([evals, times], axis=1)

    return evals


if __name__ == '__main__':
    groups = ['Weekly', 'Daily', 'Hourly', 'Quarterly', 'Monthly', 'Yearly']#, 'Hourly', 'Daily']
    lib = ['statsforecast', 'ets-r', 'statsmodels']#, 'pmdarima', 'prophet']
    evaluation = [evaluate(lib, group) for lib, group in product(lib, groups)]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation).sort_values(['dataset', 'model']).reset_index(drop=True)
    evaluation = evaluation[['dataset', 'model', 'MASE', 'time']]
    #evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(2)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation.to_csv('data/m4-evaluation.csv')
    print(evaluation.to_markdown(index=False))
