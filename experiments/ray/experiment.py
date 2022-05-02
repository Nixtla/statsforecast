import argparse
import os
from time import time

import ray
import pandas as pd
from statsforecast.utils import generate_series
from statsforecast.models import auto_arima
from statsforecast.core import StatsForecast


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Scale StatsForecast using ray')
    parser.add_argument('--ray-address')
    args = parser.parse_args()

    for length in [10_000, 100_000, 500_000, 1_000_000, 2_000_000]:
        print(f'length: {length}')
        series = generate_series(n_series=length, seed=1)

        model = StatsForecast(series, 
                              models=[auto_arima], freq='D', 
                              n_jobs=-1, 
                              ray_address=args.ray_address)
        init = time()
        forecasts = model.forecast(7)
        total_time = (time() - init) / 60
        print(f'n_series: {length} total time: {total_time}')

