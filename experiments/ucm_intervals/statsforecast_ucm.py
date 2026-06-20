import time
import fire
import pandas as pd
from multiprocessing import cpu_count
from statsforecast import StatsForecast
from statsforecast.models import UCM
from data import get_data

def find_seasonality(group: str) -> int:
    seasonality = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 7,
        'Hourly': 24,
        'Other': 1,
    }
    if group not in seasonality:
        raise ValueError(f'Group {group} not found')
    return seasonality[group]

def main(dataset: str = 'M3', group: str = 'Other') -> None:
    train, horizon, freq, _ = get_data('data/', dataset, group)

    if dataset == 'M4':
        train['ds'] = train['ds'].astype(int)
        freq = 1  # since values in ds column are integers

    season_length = find_seasonality(group)
    models = [UCM(season_length=season_length, alias='UCM-sf')]

    start = time.time()
    sf = StatsForecast(models=models, freq=freq, n_jobs=cpu_count())
    forecasts = sf.forecast(df=train, h=horizon, level=[80, 95])
    end = time.time()

    forecasts.to_csv(f'data/UCM-sf-{dataset}-{group}.csv', index=False)

    time_df = pd.DataFrame({'time': [end - start], 'model': ['UCM-sf']})
    time_df.to_csv(f'data/UCM-sf-time-{dataset}-{group}.csv', index=False)

    print(f'Dataset: {dataset} - Group: {group} with statsforecast completed in {end - start} seconds')

if __name__ == '__main__':
    fire.Fire(main)
