from itertools import product

import fire
import pandas as pd
from datasetsforecast.losses import mape, smape

from src.data import get_data


def evaluate(lib: str, model: str, dataset: str, group: str):
    y_test, horizon, freq, seasonality = get_data('data/', dataset, group, False)
    y_test = y_test['y'].values.reshape(-1, horizon)

    forecast = pd.read_csv(f'data/{lib}-forecasts-{dataset}-{group}.csv')
    y_hat = forecast[model].values.reshape(-1, horizon)

    evals = {}
    for metric in (mape, smape):
        metric_name = metric.__name__
        loss = metric(y_test, y_hat)
        evals[metric_name] = loss 

    evals = pd.DataFrame(evals, index=[f'{dataset}_{group}']).rename_axis('dataset').reset_index()
    times = pd.read_csv(f'data/{lib}-time-{dataset}-{group}.csv')
    times['model'] = model
    evals = pd.concat([evals, times], axis=1)

    return evals

def main(test: bool = False):
    if test:
        groups = ['Other']
    else:
        groups = ['Monthly', 'Yearly', 'Other', 'Quarterly']
    lib = ['StatisticalEnsemble']
    models = ['StatisticalEnsemble', 'AutoARIMA', 'CES', 'AutoETS', 'DynamicOptimizedTheta']
    datasets = ['M3']
    evaluation = [evaluate(lib, model, dataset, group) for lib, model, group in product(lib, models, groups) for dataset in datasets]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[['dataset', 'model', 'mape', 'smape', 'time']]
    evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(3)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation.to_csv('data/evaluation.csv')
    smape = evaluation.query('metric=="smape"').T
    time = evaluation.query('metric=="time"').T
    if test:
        expected_results = {
            'AutoARIMA': 4.46,
            'CES': 4.85, 
            'AutoETS': 4.35, 
            'DynamicOptimizedTheta': 4.54,
            'StatisticalEnsemble': 4.3,
        }
        expected_results = pd.Series(expected_results)
        pd.testing.assert_series_equal(
            smape.loc[expected_results.index].iloc[:, 0].astype(float),
            expected_results,
            check_names=False,
            rtol=1e-2,
            check_exact=False,
        )
        assert time.loc[lib[0]].item() < 2.


if __name__ == '__main__':
    fire.Fire(main)
