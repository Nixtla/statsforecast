from itertools import product
import pandas as pd
from datasetsforecast.m4 import M4Info, M4Evaluation


def read_data(lib:str, group:str):
    horizon = M4Info[group].horizon
    forecast = pd.read_csv(f'data/{lib}-forecasts-M4-{group}.csv')
    if lib == 'statsforecast':
        col = 'auto_arima_nixtla'
    elif lib == 'prophet':
        col = 'prophet'
    elif lib == 'pmdarima':
        col = 'auto_arima_pmdarima'
    elif lib == 'fable-arima-r':
        col = 'fable_arima_r'
    elif lib == 'forecast-arima-r':
        col = 'forecast_arima_r'
        if group == 'Daily':
            prefix = 'D'
        elif group == 'Weekly':
            prefix = 'W'
        else:
            prefix = 'M'
        forecast = pd.read_csv(f'data/{lib}-forecasts-M4-{group}.csv', sep = " ", header = None)
        forecast = forecast.assign(unique_id = lambda x: prefix +x.index.astype(str)).melt(id_vars='unique_id', var_name='ds', value_name='forecast_arima_r')
    forecast = forecast[col].values.reshape(-1, horizon)
    return forecast


def evaluate(lib: str, group: str):
    try:
        forecast = read_data(lib, group)
        evals = M4Evaluation().evaluate('data', group, forecast)
        times = pd.read_csv(f'data/{lib}-time-M4-{group}.csv')
        evals = evals.rename_axis('dataset').reset_index()
        evals = pd.concat([evals, times], axis=1)
    except:
        return None
    return evals


if __name__ == '__main__':
    groups = ['Weekly', 'Hourly', 'Daily']
    lib = ['statsforecast', 'forecast-arima-r','pmdarima', 'prophet']#, 'fable-arima-r']
    evaluation = [evaluate(lib, group) for lib, group in product(lib, groups)]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation).sort_values(['dataset', 'model']).reset_index(drop=True)
    evaluation = evaluation[['dataset', 'model', 'MASE', 'time']]
    evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(2)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation_std = evaluation.assign(
        auto_arima_nixtla = lambda x: x.auto_arima_nixtla / x.auto_arima_r,
        auto_arima_pmdarima  = lambda x: x.auto_arima_pmdarima / x.auto_arima_r,
        prophet = lambda x: x.prophet / x.auto_arima_r,
        auto_arima_r = lambda x: x.auto_arima_r / x.auto_arima_r,
        #auto_arima_fable_r = lambda x: x.auto_arima_fable_r / x.auto_arima_fable_r
    )
    evaluation.to_csv('data/M4-evaluation.csv')
    evaluation_std.to_csv('data/M4-evaluation_std.csv')
    print(evaluation.to_markdown(index=False))