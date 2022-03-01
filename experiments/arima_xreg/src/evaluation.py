import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape, rmse, smape, mae, mase


def evaluate(group: str):
    train = pd.read_csv(f'data/EPF-{group}.csv')[['unique_id', 'ds', 'y']]
    test = pd.read_csv(f'data/EPF-{group}-test.csv')[['unique_id', 'ds', 'y']]
    #prophet = pd.read_csv('data/prophet-forecasts.csv')
    pmdarima = pd.read_csv(f'data/pmdarima-forecasts-{group}.csv')
    stats = pd.read_csv(f'data/statsforecast-forecasts-{group}.csv')
    arima = np.loadtxt(f'data/arima-r-forecasts-{group}.txt')
    
    dfs = [test, stats, pmdarima]
    dfs = [df.set_index(['unique_id', 'ds']) for df in dfs]
    df_eval = pd.concat(dfs, axis=1).reset_index()
    df_eval['auto_arima_r'] = arima
    
    models = df_eval.drop(columns=['unique_id', 'ds', 'y']).columns.to_list()
    evals = {}
    y_test = df_eval['y'].values
    y_train = train['y'].values
    for model in models:
        evals[model] = {}
        y_hat = df_eval[model].values
        for metric in (mape, rmse, smape, mae, mase):
            metric_name = metric.__name__
            if metric_name == 'mase':
                loss = mase(y_test, y_hat, y_train, seasonality=24)
            else:
                loss = metric(y_test, y_hat)
            evals[model][metric_name] = loss

    evals = pd.DataFrame(evals).rename_axis('metric').reset_index()
    models_str = ['pmdarima', 'statsforecast', 'arima-r']
    times = [pd.read_csv(f'data/{model}-time-{group}.csv') for model in models_str]
    times = pd.concat(times)
    times['time'] /= 60 #minutes
    times = times.set_index('model').T.rename_axis('metric').reset_index()
    evals = pd.concat([evals, times])

    evals.insert(0, 'dataset', group)
    evals = evals.query('metric in ["mase", "time"]')

    return evals

if __name__=="__main__":
    groups = ['NP', 'PJM', 'BE', 'FR', 'DE']
    evals = [evaluate(group) for group in groups]
    evals = pd.concat(evals)
    print(evals.groupby('metric').mean().round(2).to_markdown())

