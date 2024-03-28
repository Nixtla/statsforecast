from itertools import product
import fire
import pandas as pd
from utilsforecast.evaluation import evaluate 
from utilsforecast.losses import mae, rmse, smape 

from data import get_data

def accuracy(model: str, dataset: str, group: str):
    # Note PY-TBATS is only available for M3 dataset
    y_test, _, _, _ = get_data('data/', dataset, group, False)
    y_test['id'] = y_test.groupby('unique_id').cumcount()+1 # create id column to merge with R forecasts 
    y_test.drop(columns=['ds'], inplace=True)

    tbats = pd.read_csv(f'data/AutoTBATS-{dataset}-{group}.csv') 
    snaive = pd.read_csv(f'data/SeasonalNaive-{dataset}-{group}.csv')
    stats_fcst = tbats.merge(snaive, on=['unique_id', 'ds'])
    stats_fcst['id'] = stats_fcst.groupby('unique_id').cumcount()+1 

    r_fcst = pd.read_csv(f'data/R-forecasts-{dataset}-{group}.csv')

    if dataset == 'M3': 
        py_fcst = pd.read_csv(f'data/PY-TBATS-forecasts-{dataset}-{group}.csv')
        py_fcst['id'] = py_fcst.groupby('unique_id').cumcount()+1 
        py_fcst.rename(columns={'y': 'PY-TBATS'}, inplace=True)
        forecasts = stats_fcst.merge(r_fcst, on=['unique_id', 'id'])
        forecasts = forecasts.merge(py_fcst, on=['unique_id', 'id'])
    else: 
        forecasts = stats_fcst.merge(r_fcst, on=['unique_id', 'id'])

    # Validation 
    if forecasts.shape[0] != stats_fcst.shape[0] or forecasts.shape[0] != r_fcst.shape[0]:
        raise ValueError(f'The number of forecasts is not the same for AutoTBATS and R-TBATS for {dataset} {group}')
    
    predictions = forecasts.merge(y_test, on=['unique_id', 'id'])
    predictions.drop(columns=['id'], inplace=True)   
    predictions = predictions[['unique_id', 'ds', 'y', 'AutoTBATS', 'R-TBATS', 'SeasonalNaive'] + (['PY-TBATS'] if dataset == 'M3' else [])]

    metrics = [mae, rmse, smape]
    evaluation = evaluate(
        predictions,
        metrics=metrics, 
        models=[model]
    )
    evals = evaluation.groupby('metric').mean(numeric_only=True).reset_index()
    evals = pd.melt(evals, id_vars=['metric'], value_vars=[model], var_name='model', value_name='value')
    evals = evals.pivot(index='model', columns='metric', values='value').reset_index()
    evals['dataset'] = f'{dataset}_{group}'

    # Add total execution time 
    if model == 'AutoTBATS':
        time = pd.read_csv(f'data/AutoTBATS-time-{dataset}-{group}.csv')
        time['model'] = model 
    elif model == 'SeasonalNaive': 
        time = pd.read_csv(f'data/SeasonalNaive-time-{dataset}-{group}.csv')
        time['model'] = model
    elif model == 'R-TBATS': 
        time = pd.read_csv(f'data/R-time-{dataset}-{group}.csv')
        time['model'] = 'R-TBATS'
    elif dataset == 'M3':
        # add PY-TBATS time when using M3 dataset
        time = pd.read_csv(f'data/PY-TBATS-time-{dataset}-{group}.csv')
        time['model'] = 'PY-TBATS'

    evals = pd.merge(evals, time, on=['model'], how='left')

    return evals

def main(dataset: str = 'M3'):
    if dataset == 'M3':
        groups = ['Yearly', 'Quarterly', 'Monthly', 'Other']
        models = ['AutoTBATS', 'R-TBATS', 'PY-TBATS', 'SeasonalNaive']
    elif dataset == 'M4':
        groups = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
        models = ['AutoTBATS', 'R-TBATS', 'SeasonalNaive']
    else: 
        raise ValueError(f'Dataset {dataset} not found')
    evaluation = [accuracy(model, dataset, group) for model, group in product(models, groups)]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation)
    evaluation = evaluation[['dataset', 'model', 'mae', 'rmse', 'smape', 'time']]
    evaluation['smape'] = evaluation['smape'] * 100
    evaluation[['mae', 'rmse', 'smape']] = evaluation[['mae', 'rmse', 'smape']].round(2)
    evaluation = evaluation.sort_values(by=['dataset', 'model'])
    evaluation['time'] = evaluation['time'] / 60 # convert to minutes
    evaluation.to_csv(f'data/evaluation-{dataset}.csv')
    print(evaluation)

if __name__ == '__main__':
    fire.Fire(main)
