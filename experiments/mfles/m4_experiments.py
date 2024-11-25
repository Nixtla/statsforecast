import argparse
import os
import random
from functools import partial
from pathlib import Path

os.environ['NIXTLA_NUMBA_CACHE'] = '1'

import pandas as pd
from datasetsforecast.m4 import M4, M4Info

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoMFLES,
    AutoTBATS,
    DynamicOptimizedTheta,
    SeasonalNaive,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group')
    args = parser.parse_args()

    # data
    seasonality_overrides = {
        "Daily": [7],
        "Hourly": [24, 24*7],
        "Weekly": [52],
    }
    group = args.group.capitalize()
    df, *_ = M4.load("data", group)
    df['ds'] = df['ds'].astype('int64')
    info = M4Info[group]
    h = info.horizon
    season_length = [info.seasonality]
    if group in seasonality_overrides:
        season_length = seasonality_overrides[group]
    valid = df.groupby("unique_id").tail(h)
    train = df.drop(valid.index)
    print(f'Running {group}. season_length: {season_length}')

    # forecast
    sf = StatsForecast(
        models=[
            AutoARIMA(season_length=season_length[0]),
            AutoETS(season_length=season_length[0]),
            AutoMFLES(test_size=h, season_length=season_length, n_windows=3),
            AutoTBATS(season_length=season_length),
            DynamicOptimizedTheta(season_length=season_length[0]),
            SeasonalNaive(season_length=season_length[0]),
        ],
        freq=1,
        n_jobs=-1,
        verbose=True,
    )
    preds = sf.forecast(df=train, h=h)
    res = preds.merge(valid, on=['unique_id', 'ds'])

    # save results
    results_path = Path('results') / group.lower()
    results_path.mkdir(exist_ok=True)
    res = preds.merge(valid, on=['unique_id', 'ds'])
    res.to_parquet(results_path / 'valid.parquet', index=False)
    train.to_parquet(results_path / 'train.parquet', index=False)
    times = pd.Series(sf.forecast_times_).reset_index()
    times.columns = ['model', 'time']
    times.to_csv(results_path / 'times.csv', index=False)
