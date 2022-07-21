import argparse
import os
from time import time

import ray
import pandas as pd
from statsforecast.utils import generate_series
from statsforecast.models import ets
from statsforecast.core import StatsForecast


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Scale StatsForecast using ray')
    parser.add_argument('--ray-address')
    parser.add_argument('--seasonality', type=int)
    args = parser.parse_args()
    ray_address = f'ray://{args.ray_address}:10001'
    models = [(ets, args.seasonality)] if args.seasonality else [ets] 

    for length in [1_000_000, 2_000_000]:
        print(f'length: {length}')
        series = generate_series(n_series=length, seed=1, equal_ends=True)
        # add constant to avoid numerical errors
        # in production settings we simply remove this constant
        # from the forecasts
        series['y'] += 10

        model = StatsForecast(series, 
                              models=models, 
                              freq='D', 
                              n_jobs=-1, 
                              ray_address=ray_address)
        init = time()
        forecasts = model.forecast(7)
        total_time = (time() - init) / 60
        print(f'n_series: {length} total time: {total_time}')

