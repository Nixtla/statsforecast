import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import fire
import warnings
import concurrent.futures
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from statsmodels.tsa.statespace.structural import UnobservedComponents
from data import get_data
from statsforecast_ucm import find_seasonality

warnings.simplefilter("ignore")

def generate_fcst(vals):
    unique_id, y, horizon, season_length = vals
    # Equivalent to the statsforecast UCM: local linear trend + seasonal + irregular.
    kwargs = dict(level="local linear trend", irregular=True)
    if season_length > 1:
        kwargs.update(seasonal=season_length, stochastic_seasonal=True)
    res = UnobservedComponents(y, **kwargs).fit(disp=False)
    pred = res.get_forecast(horizon)
    mean = np.asarray(pred.predicted_mean)
    lo80, hi80 = np.asarray(pred.conf_int(alpha=0.20)).T
    lo95, hi95 = np.asarray(pred.conf_int(alpha=0.05)).T
    return unique_id, mean, lo80, hi80, lo95, hi95

def main(dataset: str = 'M4', group: str = 'Monthly', n_workers: int = None) -> None:
    train, horizon, _, _ = get_data('data/', dataset, group)
    season_length = find_seasonality(group)

    vals_list = [
        (uid, grp['y'].values, horizon, season_length)
        for uid, grp in train.groupby('unique_id', sort=False)
    ]
    n_series = len(vals_list)

    if n_workers is None:
        n_workers = cpu_count()
    n_workers = min(n_workers, n_series)

    chunksize = min(256, max(1, n_series // (n_workers * 8)))

    ids, means, lo80s, hi80s, lo95s, hi95s = [], [], [], [], [], []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for uid, mean, lo80, hi80, lo95, hi95 in executor.map(generate_fcst, vals_list, chunksize=chunksize):
            ids.append(uid)
            means.append(mean)
            lo80s.append(lo80)
            hi80s.append(hi80)
            lo95s.append(lo95)
            hi95s.append(hi95)
    end = time.time()

    forecasts = pd.DataFrame({
        'unique_id': np.repeat(ids, horizon),
        'UCM-sm': np.concatenate(means),
        'UCM-sm-lo-80': np.concatenate(lo80s),
        'UCM-sm-hi-80': np.concatenate(hi80s),
        'UCM-sm-lo-95': np.concatenate(lo95s),
        'UCM-sm-hi-95': np.concatenate(hi95s),
    })

    forecasts.to_csv(f'data/UCM-sm-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['UCM-sm']})
    time_df.to_csv(f'data/UCM-sm-time-{dataset}-{group}.csv', index=False)

    print(f'Dataset: {dataset} - Group: {group} with statsmodels completed in {end - start} seconds')

if __name__ == '__main__':
    fire.Fire(main)
