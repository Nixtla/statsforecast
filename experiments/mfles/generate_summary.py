from functools import partial
from pathlib import Path

import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, mape, rmse, mae, mase


def generate_metrics(path: Path) -> str:
    seasonalities = {
        'hourly': 24,
        'daily': 7,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4,
        'yearly': 1,
    }
    fmts = {
        'mase': '{:.2f}',
        'rmse': '{:,.1f}',
        'smape': '{:.1%}',
        'mape': '{:.1%}',
        'mae': '{:,.1f}',
    }
    season_length = seasonalities[path.name]
    pmase = partial(mase, seasonality=season_length)

    train = pd.read_parquet(path / 'train.parquet')
    valid = pd.read_parquet(path / 'valid.parquet')
    eval_res = evaluate(
        valid,
        train_df=train,
        metrics=[smape, mape, rmse, mae, pmase],
    )
    summary = eval_res.drop(columns='unique_id').groupby('metric').mean()
    formatted = {}
    for metric in summary.index:
        row = summary.loc[metric]
        best = row.idxmin()
        fmt = fmts[metric]
        row = row.map(fmt.format)
        row[best] = '**' + row[best] + '**'
        formatted[metric] = row
    out_cols = [c for c in summary.columns if c != 'AutoARIMA']
    if 'AutoARIMA' in summary.columns:
        out_cols.append('AutoARIMA')
    return pd.DataFrame(formatted).T[out_cols].to_markdown()


def generate_times(path: Path) -> str:
    df = pd.read_csv(path / 'times.csv')
    df['time'] /= 60
    df = df.sort_values('time')
    df = df.rename(columns={'time': 'CPU time (min)'})
    return df.to_markdown(index=False, floatfmt=',.0f')


if __name__ == '__main__':
    with open('summary.md', 'wt') as f:
        for path in Path('results').iterdir():
            f.write(f'# {path.name.capitalize()}')
            f.write('\n## Metrics\n')
            f.write(generate_metrics(path))
            f.write('\n\n## Times\n')
            f.write(generate_times(path))
            f.write('\n')
