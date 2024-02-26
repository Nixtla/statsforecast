import os
import time

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoTBATS

os.environ['NIXTLA_ID_AS_COL'] = '1'

data = pd.read_csv("https://datasets-nixtla.s3.amazonaws.com/m4-hourly.csv")
horizon = 24
train_mask = data['ds'] <= data.groupby('unique_id').transform('size') - horizon
train = data[train_mask].reset_index(drop=True)
valid = data[~train_mask].reset_index(drop=True)
start = time.perf_counter()
sf = StatsForecast(
    models=[AutoTBATS(seasonal_periods=[24, 24 * 7])],
    freq=1,
    n_jobs=-1,
)
preds = sf.forecast(df=train, h=horizon)
time_taken = time.perf_counter() - start
res = valid.merge(preds, on=['unique_id', 'ds'], how='left')
res['smape'] = 2 * abs(res['y'] - res['AutoTBATS']) / (abs(res['y']) + abs(res['AutoTBATS']))
print(f'Time taken (minutes) {time_taken / 60:.1f}')
print(f"Average SMAPE: {res.groupby('unique_id')['smape'].mean().mean():.1%}")
