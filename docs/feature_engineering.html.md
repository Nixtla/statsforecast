---
description: Generate features for downstream models
output-file: feature_engineering.html
title: Feature engineering
---

::: statsforecast.feature_engineering.mstl_decomposition

```python
import pandas as pd
from fastcore.test import test_fail
from utilsforecast.losses import smape

from statsforecast.models import Naive
from statsforecast.utils import generate_series

series = generate_series(10, freq='D')
series['unique_id'] = series['unique_id'].astype('int64')
```

```python
horizon = 14
model = MSTL(season_length=7)
series = series.sample(frac=1.0)
train_df, X_df = mstl_decomposition(series, model, 'D', horizon)
```

```python
series_pl = generate_series(10, freq='D', engine='polars')
series_pl = series_pl.with_columns(unique_id=pl.col('unique_id').cast(pl.Int64))
train_df_pl, X_df_pl = mstl_decomposition(series_pl, model, '1d', horizon)
```

```python
pd.testing.assert_series_equal(
    train_df.groupby('unique_id')['ds'].max() + pd.offsets.Day(),
    X_df.groupby('unique_id')['ds'].min()
)
assert X_df.shape[0] == train_df['unique_id'].nunique() * horizon
pd.testing.assert_frame_equal(train_df, train_df_pl.to_pandas())
pd.testing.assert_frame_equal(X_df, X_df_pl.to_pandas())
with_estimate = train_df_pl.with_columns(estimate=pl.col('trend') + pl.col('seasonal'))
assert smape(with_estimate, models=['estimate'])['estimate'].mean() < 0.1
```

```python
model = MSTL(season_length=[7, 28])
train_df, X_df = mstl_decomposition(series, model, 'D', horizon)
assert train_df.columns.intersection(X_df.columns).tolist() == ['unique_id', 'ds', 'trend', 'seasonal7', 'seasonal28']
```
