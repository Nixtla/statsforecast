---
description: >-
  Methods for Fit, Predict, Forecast (fast), Cross Validation and plotting
output-file: core.html
title: Core Methods
---

The core methods of `StatsForecast` provide a comprehensive interface for fitting, predicting, forecasting, and evaluating statistical forecasting models on large sets of time series.

## Overview

The main methods include:

- `StatsForecast.fit` - Fit statistical models
- `StatsForecast.predict` - Predict using fitted models
- `StatsForecast.forecast` - Memory-efficient predictions without storing models
- `StatsForecast.cross_validation` - Temporal cross-validation
- `StatsForecast.plot` - Visualization of forecasts and historical data

## StatsForecast Class

::: statsforecast.core.StatsForecast
    options:
      show_source: true
      heading_level: 3
      members:
        - __init__
        - fit
        - predict
        - fit_predict
        - forecast
        - forecast_fitted_values
        - cross_validation
        - cross_validation_fitted_values
        - plot
        - save
        - load

## Usage Examples

### Basic Forecasting

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive
from statsforecast.utils import generate_series

# Generate example data
panel_df = generate_series(n_series=9, equal_ends=False, engine='pandas')

# Instantiate StatsForecast class
fcst = StatsForecast(
    models=[AutoARIMA(), Naive()],
    freq='D',
    n_jobs=1,
    verbose=True
)

# Efficiently predict
fcsts_df = fcst.forecast(df=panel_df, h=4, fitted=True)
```

### Cross-Validation

```python
from statsforecast import StatsForecast
from statsforecast.models import Naive
from statsforecast.utils import AirPassengersDF as panel_df

# Instantiate StatsForecast class
fcst = StatsForecast(
    models=[Naive()],
    freq='D',
    n_jobs=1,
    verbose=True
)

# Perform cross-validation
cv_df = fcst.cross_validation(df=panel_df, h=14, n_windows=2)
```

### Prediction Intervals

```python
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA
from statsforecast.utils import AirPassengers as ap

# Prepare data
ap_df = pd.DataFrame({'ds': np.arange(ap.size), 'y': ap})
ap_df['unique_id'] = 0

# Forecast with prediction intervals
sf = StatsForecast(
    models=[
        SeasonalNaive(season_length=12),
        AutoARIMA(season_length=12)
    ],
    freq=1,
    n_jobs=1
)
ap_ci = sf.forecast(df=ap_df, h=12, level=(80, 95))

# Plot with confidence intervals
sf.plot(ap_df, ap_ci, level=[80], engine="matplotlib")
```

### Conformal Prediction Intervals

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import ConformalIntervals

sf = StatsForecast(
    models=[
        AutoARIMA(season_length=12),
        AutoARIMA(
            season_length=12,
            prediction_intervals=ConformalIntervals(n_windows=2, h=12),
            alias='ConformalAutoARIMA'
        ),
    ],
    freq=1,
    n_jobs=1
)
ap_ci = sf.forecast(df=ap_df, h=12, level=(80, 95))
```

## Advanced Features

### Integer Datestamps

The `StatsForecast` class can work with integer datestamps instead of datetime objects:

```python
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage
from statsforecast.utils import AirPassengers as ap
import pandas as pd
import numpy as np

# Create dataframe with integer datestamps
int_ds_df = pd.DataFrame({'ds': np.arange(1, len(ap) + 1), 'y': ap})
int_ds_df.insert(0, 'unique_id', 'AirPassengers')

# Use freq=1 for integer datestamps
fcst = StatsForecast(models=[HistoricAverage()], freq=1)
forecast = fcst.forecast(df=int_ds_df, h=7)
```

### External Regressors

Every column after `y` is considered an external regressor and will be passed to models that support them:

```python
from statsforecast import StatsForecast
from statsforecast.utils import generate_series
import pandas as pd

# Create data with external regressors
series_xreg = generate_series(10_000, equal_ends=True)
series_xreg['intercept'] = 1
series_xreg['dayofweek'] = series_xreg['ds'].dt.dayofweek
series_xreg = pd.get_dummies(series_xreg, columns=['dayofweek'], drop_first=True)

# Split train/validation
dates = sorted(series_xreg['ds'].unique())
valid_start = dates[-14]
train_mask = series_xreg['ds'] < valid_start
series_train = series_xreg[train_mask]
series_valid = series_xreg[~train_mask]
X_valid = series_valid.drop(columns=['y'])

# Forecast with external regressors
fcst = StatsForecast(models=[your_model], freq='D')
xreg_res = fcst.forecast(df=series_train, h=14, X_df=X_valid)
```

## Distributed Computing

The `StatsForecast` class offers parallelization utilities with Dask, Spark and Ray backends for distributed computing. See the [distributed computing examples](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray) for more information.
