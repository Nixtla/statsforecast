# statsforecast
> Forecasting using statistical models


[![CI](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml/badge.svg?branch=refactor)](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml)
![PyPI](https://img.shields.io/pypi/v/statsforecast?color=blue)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/statsforecast)
[![License](https://img.shields.io/github/license/Nixtla/statsforecast)](https://github.com/Nixtla/statsforecast/tree/refactor/LICENSE)

## Install
`pip install statsforecast`

## How to use

```python
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

from statsforecast import StatsForecast
from statsforecast.models import random_walk_with_drift, seasonal_naive, ses
```

```python
def display_df(df):
    display(Markdown(df.to_markdown()))
```

```python
rng = np.random.RandomState(0)
serie1 = np.arange(1, 8)[np.arange(100) % 7] + rng.randint(-1, 2, size=100)
serie2 = np.arange(100) + rng.rand(100)
series = pd.DataFrame(
    {
        'ds': pd.date_range('2000-01-01', periods=serie1.size + serie2.size, freq='D'),
        'y': np.hstack([serie1, serie2]),
    },
    index=pd.Index([0] * serie1.size + [1] * serie2.size, name='unique_id')
)
display_df(pd.concat([series.head(), series.tail()]))
```


|   unique_id | ds                  |       y |
|------------:|:--------------------|--------:|
|           0 | 2000-01-01 00:00:00 |  0      |
|           0 | 2000-01-02 00:00:00 |  2      |
|           0 | 2000-01-03 00:00:00 |  2      |
|           0 | 2000-01-04 00:00:00 |  4      |
|           0 | 2000-01-05 00:00:00 |  5      |
|           1 | 2000-07-14 00:00:00 | 95.7649 |
|           1 | 2000-07-15 00:00:00 | 96.9441 |
|           1 | 2000-07-16 00:00:00 | 97.75   |
|           1 | 2000-07-17 00:00:00 | 98.3394 |
|           1 | 2000-07-18 00:00:00 | 99.4895 |


```python
fcst = StatsForecast(series, models=[random_walk_with_drift, (seasonal_naive, 7), (ses, 0.1)], freq='D')
forecasts = fcst.forecast(5)
display_df(forecasts)
```

    2021-11-23 19:14:48 statsforecast.core INFO: Computing forecasts
    2021-11-23 19:14:49 statsforecast.core INFO: Computed forecasts for random_walk_with_drift.
    2021-11-23 19:14:49 statsforecast.core INFO: Computed forecasts for seasonal_naive_season_length-7.
    2021-11-23 19:14:49 statsforecast.core INFO: Computed forecasts for ses_alpha-0.1.



|   unique_id | ds                  |   random_walk_with_drift |   seasonal_naive_season_length-7 |   ses_alpha-0.1 |
|------------:|:--------------------|-------------------------:|---------------------------------:|----------------:|
|           0 | 2000-04-10 00:00:00 |                  3.0303  |                           3      |         3.85506 |
|           0 | 2000-04-11 00:00:00 |                  3.06061 |                           5      |         3.85506 |
|           0 | 2000-04-12 00:00:00 |                  3.09091 |                           4      |         3.85506 |
|           0 | 2000-04-13 00:00:00 |                  3.12121 |                           7      |         3.85506 |
|           0 | 2000-04-14 00:00:00 |                  3.15152 |                           6      |         3.85506 |
|           1 | 2000-07-19 00:00:00 |                100.489   |                          93.0166 |        90.4709  |
|           1 | 2000-07-20 00:00:00 |                101.489   |                          94.2307 |        90.4709  |
|           1 | 2000-07-21 00:00:00 |                102.489   |                          95.7649 |        90.4709  |
|           1 | 2000-07-22 00:00:00 |                103.489   |                          96.9441 |        90.4709  |
|           1 | 2000-07-23 00:00:00 |                104.489   |                          97.75   |        90.4709  |

