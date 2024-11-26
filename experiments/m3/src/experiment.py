import os
import time
from multiprocessing import cpu_count
os.environ['NIXTLA_NUMBA_RELEASE_GIL'] = '1'
os.environ['NIXTLA_NUMBA_CACHE'] = '1'

import fire
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
        AutoTheta, AutoETS, AutoCES,AutoARIMA,
        DynamicOptimizedTheta
)
from statsforecast.utils import AirPassengers as ap

from src.data import get_data


def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, seasonality = get_data('data/', dataset, group)
    models = [
        AutoETS(season_length=seasonality),
        AutoCES(season_length=seasonality),
        AutoARIMA(season_length=seasonality),
        DynamicOptimizedTheta(season_length=seasonality),
    ]

    start = time.time()
    fcst = StatsForecast(models=models, freq=freq, n_jobs=cpu_count())
    forecasts = fcst.forecast(df=train, h=horizon)
    end = time.time()
    print(end - start)

    forecasts['StatisticalEnsemble'] = forecasts.set_index(['unique_id', 'ds']).median(axis=1).values
    forecasts.to_csv(f'data/StatisticalEnsemble-forecasts-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['StatisticalEnsemble']})
    time_df.to_csv(f'data/StatisticalEnsemble-time-{dataset}-{group}.csv', index=False)


if __name__ == '__main__':
    AutoTheta(season_length=12).forecast(ap.astype(np.float32), h=12)
    fire.Fire(main)
