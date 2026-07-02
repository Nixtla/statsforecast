import fire
import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape
from data import get_data

models = ['UCM-sf', 'UCM-sm']

def accuracy(dataset: str, group: str):
    y_test, _, _, _ = get_data('data/', dataset, group, False)
    # Positional id to merge forecasts with the test set (robust to differing
    # `ds` between datasets/libraries).
    y_test['id'] = y_test.groupby('unique_id').cumcount() + 1
    y_test = y_test.drop(columns=['ds'])

    sf = pd.read_csv(f'data/UCM-sf-{dataset}-{group}.csv')
    sf['id'] = sf.groupby('unique_id').cumcount() + 1
    sm = pd.read_csv(f'data/UCM-sm-{dataset}-{group}.csv')
    sm['id'] = sm.groupby('unique_id').cumcount() + 1

    forecasts = sf.merge(sm[['unique_id', 'id', 'UCM-sm']], on=['unique_id', 'id'])
    predictions = forecasts.merge(y_test, on=['unique_id', 'id'])
    predictions = predictions[['unique_id', 'ds', 'y'] + models]

    evaluation = evaluate(predictions, metrics=[mae, rmse, smape], models=models)
    evals = evaluation.groupby('metric').mean(numeric_only=True).reset_index()
    evals = pd.melt(
        evals, id_vars=['metric'], value_vars=models, var_name='model', value_name='value'
    )
    evals = evals.pivot(index='model', columns='metric', values='value').reset_index()
    evals['dataset'] = f'{dataset}_{group}'

    times = pd.concat([
        pd.read_csv(f'data/UCM-sf-time-{dataset}-{group}.csv'),
        pd.read_csv(f'data/UCM-sm-time-{dataset}-{group}.csv'),
    ])
    evals = evals.merge(times, on='model', how='left')
    return evals

def main(dataset: str = 'M3'):
    if dataset == 'M3':
        groups = ['Yearly', 'Quarterly', 'Monthly', 'Other']
    elif dataset == 'M4':
        groups = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    else:
        raise ValueError(f'Dataset {dataset} not found')

    evaluation = pd.concat([accuracy(dataset, group) for group in groups])
    evaluation = evaluation[['dataset', 'model', 'mae', 'rmse', 'smape', 'time']]
    evaluation['smape'] = evaluation['smape'] * 100
    evaluation['time'] = evaluation['time'] / 60
    evaluation[['mae', 'rmse', 'smape', 'time']] = evaluation[['mae', 'rmse', 'smape', 'time']].round(3)
    evaluation = evaluation.sort_values(by=['dataset', 'model'])
    evaluation.to_csv(f'data/evaluation-{dataset}.csv', index=False)
    print(evaluation.to_markdown(index=False))

if __name__ == '__main__':
    fire.Fire(main)
