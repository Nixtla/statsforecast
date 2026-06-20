import fire
import pandas as pd
from typing import List
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import scaled_crps, coverage
from data import get_data

models = ['UCM-sf', 'UCM-sm']
level = [80, 95]
interval_cols = [f'{m}-{side}-{lvl}' for m in models for side in ('lo', 'hi') for lvl in level]

def winkler(
    df,
    models: List[str],
    level: int,
    id_col: str = 'unique_id',
    target_col: str = 'y',
    cutoff_col: str = 'cutoff',
):
    alpha = 1 - level / 100
    y = df[target_col].to_numpy()
    out = df[[id_col]].copy()
    for m in models:
        lo = df[f'{m}-lo-{level}'].to_numpy()
        hi = df[f'{m}-hi-{level}'].to_numpy()
        out[m] = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return out.groupby(id_col, observed=True).mean().reset_index()

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

    sm_cols = ['unique_id', 'id', 'UCM-sm'] + [c for c in interval_cols if c.startswith('UCM-sm')]
    forecasts = sf.merge(sm[sm_cols], on=['unique_id', 'id'])
    predictions = forecasts.merge(y_test, on=['unique_id', 'id'])
    predictions = predictions[['unique_id', 'ds', 'y'] + models + interval_cols]

    evaluation = evaluate(
        predictions, metrics=[scaled_crps, winkler, coverage], models=models, level=level
    )
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

    metrics = ['scaled_crps', 'winkler_level80', 'winkler_level95', 'coverage_level80', 'coverage_level95']
    evaluation = pd.concat([accuracy(dataset, group) for group in groups])
    evaluation = evaluation[['dataset', 'model'] + metrics + ['time']]
    evaluation['time'] = evaluation['time'] / 60
    evaluation[metrics + ['time']] = evaluation[metrics + ['time']].round(4)
    evaluation = evaluation.sort_values(by=['dataset', 'model'])
    evaluation.to_csv(f'data/evaluation-{dataset}.csv', index=False)
    print(evaluation.to_markdown(index=False))

if __name__ == '__main__':
    fire.Fire(main)
