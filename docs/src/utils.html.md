---
description: >-
  The `core.StatsForecast` class allows you to efficiently fit multiple
  `StatsForecast` models for large sets of time series. It operates with pandas
  DataFrame `df` that identifies individual series and datestamps with the
  `unique_id` and `ds` columns, and the `y` column denotes the target time
  series variable. To assist development, we declare useful datasets that we use
  throughout all `StatsForecast`'s unit tests.
title: Utils
---

## 1. Synthetic Panel Data

::: statsforecast.utils.generate_series

```python
synthetic_panel = generate_series(n_series=2)
synthetic_panel.groupby('unique_id', observed=True).head(4)
```

## 2. AirPassengers Data

The classic Box & Jenkins airline data. Monthly totals of international
airline passengers, 1949 to 1960.

It has been used as a reference on several forecasting libraries, since
it is a series that shows clear trends and seasonalities it offers a
nice opportunity to quickly showcase a model’s predictions performance.

```python
from statsforecast.utils import AirPassengersDF

#We are going to plot the ARIMA predictions, and the prediction intervals.
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = AirPassengersDF.set_index('ds')

plot_df[['y']].plot(ax=ax, linewidth=2)
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```
