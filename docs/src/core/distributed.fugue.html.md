---
output-file: distributed.fugue.html
title: Fugue Backend
---

The `FugueBackend` class enables distributed computation for StatsForecast using [Fugue](https://github.com/fugue-project/fugue), which provides a unified interface for Spark, Dask, and Ray backends without requiring code rewrites.

## Overview

With FugueBackend, you can:

- Distribute forecasting and cross-validation across clusters
- Switch between Spark, Dask, and Ray without changing your code
- Scale to large datasets with parallel processing
- Maintain the same API as the standard StatsForecast interface

## API Reference

::: statsforecast.distributed.fugue.FugueBackend
    options:
      show_source: true
      heading_level: 3
      members:
        - __init__
        - forecast
        - cross_validation

## Quick Start

### Basic Usage with Spark

```python
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from statsforecast.utils import generate_series
from pyspark.sql import SparkSession

# Generate example data
n_series = 4
horizon = 7
series = generate_series(n_series)

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Convert unique_id to string and create Spark DataFrame
series['unique_id'] = series['unique_id'].astype(str)
sdf = spark.createDataFrame(series)

# Use StatsForecast with Spark DataFrame (automatically uses FugueBackend)
sf = StatsForecast(
    models=[AutoETS(season_length=7)],
    freq='D',
)

# Returns a Spark DataFrame
results = sf.cross_validation(
    df=sdf,
    h=horizon,
    step_size=24,
    n_windows=2,
    level=[90]
)
results.show()
```

### Basic Forecasting

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsforecast.utils import generate_series

# Generate data
series = generate_series(n_series=4)

# Standard usage (pandas/polars)
sf = StatsForecast(
    models=[AutoETS(season_length=7)],
    freq='D',
)

# Forecast with pandas DataFrame
sf.cross_validation(
    df=series,
    h=7,
    step_size=24,
    n_windows=2,
    level=[90]
).head()
```

## Dask Distributed Example

Here's a complete example using Dask for distributed predictions:

```python
import dask.dataframe as dd
from dask.distributed import Client
from fugue_dask import DaskExecutionEngine
from statsforecast import StatsForecast
from statsforecast.models import Naive
from statsforecast.utils import generate_series

# Generate synthetic panel data
df = generate_series(10)
df['unique_id'] = df['unique_id'].astype(str)
df = dd.from_pandas(df, npartitions=10)

# Instantiate Dask client and execution engine
dask_client = Client()
engine = DaskExecutionEngine(dask_client=dask_client)

# Create StatsForecast instance
sf = StatsForecast(models=[Naive()], freq='D')
```

### Distributed Forecast

The FugueBackend automatically handles distributed forecasting when you pass a Dask/Spark/Ray DataFrame:

```python
# Distributed predictions
forecast_df = sf.forecast(df=df, h=12).compute()

# With fitted values
sf = StatsForecast(models=[Naive()], freq='D')
forecast_df = sf.forecast(df=df, h=12, fitted=True).compute()
fitted_df = sf.forecast_fitted_values().compute()
```

### Distributed Cross-Validation

Perform distributed temporal cross-validation across your cluster:

```python
# Distributed cross-validation
cv_results = sf.cross_validation(
    df=df,
    h=12,
    n_windows=2
).compute()
```

## How It Works

1. **Automatic Detection**: When you pass a Spark, Dask, or Ray DataFrame to StatsForecast methods, the FugueBackend is automatically used.

2. **Data Partitioning**: Data is partitioned by `unique_id`, allowing parallel processing across different time series.

3. **Distributed Execution**: Each partition is processed independently using the standard StatsForecast logic.

4. **Result Aggregation**: Results are collected and returned in the same format as the input (Spark/Dask/Ray DataFrame).

## Supported Backends

- **Apache Spark**: For large-scale distributed processing
- **Dask**: For flexible distributed computing with Python
- **Ray**: For modern distributed machine learning workloads

## Notes

- Ensure your cluster has sufficient resources for the number of time series and models
- The `unique_id` column should be string type for distributed operations
- Use `.compute()` on Dask DataFrames to materialize results
- Use `.show()` or `.collect()` on Spark DataFrames to view results

## See Also

- [Core StatsForecast Methods](core.html)
- [Distributed Computing Examples](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray)
- [Fugue Documentation](https://fugue-tutorials.readthedocs.io/)
