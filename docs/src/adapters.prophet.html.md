---
description: >-
  In 2017, Facebook open-sourced
  [Prophet](https://peerj.com/preprints/3190.pdf), with the promise of providing
  experts and non-experts the possibility of producing high-quality predictions.
  The forecasting community heavily adopted the solution, reaching millions of
  accumulated downloads. It became evident that its [quality is
  shadowed](https://www.reddit.com/r/MachineLearning/comments/wqrw8x/d_fool_me_once_shame_on_you_fool_me_twice_shame/)
  by simpler well-proven methods. This effort aims to provide an alternative to
  overcome the Prophet's memory.<br/><br/><div align="center">"It is important to
  note that false prophets sometimes prophesied accurately, ... "
  <br/>(Deuteronomy 13:2,5) </div>
output-file: adapters.prophet.html
title: Replace FB-Prophet
---

# 1. AutoARIMA Adapter

## AutoArimaProphet

::: statsforecast.adapters.prophet.AutoARIMAProphet
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
      heading_level: 3
      show_root_heading: true
      show_source: true

### Quick Start

In this example, we revisit the time series of the log daily page views
for the Wikipedia page for [Peyton
Manning](https://en.wikipedia.org/wiki/Peyton_Manning). The dataset was
scraped this data using the
[WikipediaTrend](https://cran.r-project.org/web/packages/wikipediatrend/index.html)
package in R.

The Peyton Manning dataset was selected to illustrate Prophet’s
features, like multiple seasonality, changing growth rates, and the
ability to model special days (such as Manning’s playoff and SuperBowl
appearances). The original CSV is available
[here](https://github.com/facebook/prophet/blob/main/examples/example_wp_log_peyton_manning.csv).

Here we show that
[`AutoARIMA`](https://Nixtla.github.io/statsforecast/src/core/models.html#autoarima)
can improve performance by borrowing the `Prophet`’s feature
preprocessing.

### Inputs

The
[`AutoARIMAProphet`](https://Nixtla.github.io/statsforecast/src/adapters.prophet.html#autoarimaprophet)
adapter uses `Prophet`‘s inputs, a pandas dataframe with two columns:
`ds` and `y`. The `ds` (datestamp) column should be of a format expected
by Pandas, ideally ’YYYY-MM-DD’ for a date or ‘YYYY-MM-DD HH:MM:SS’ for
a timestamp. The `y` column must be numeric, and represents the
measurement we wish to forecast.

```python
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()
```

## 2.1 Univariate Prophet

Here we forecast with `Prophet` without external regressors. We first
instantiate a new `Prophet` object, and define its forecasting procedure
into its constructor. After that a classic sklearn `fit` and `predict`
is used to obtain the predictions.

```python
m = Prophet(daily_seasonality=False)
m.fit(df)
future = m.make_future_dataframe(365)
forecast = m.predict(future)
```

```python
fig = m.plot(forecast)
```

Here we forecast with
[`AutoARIMAProphet`](https://Nixtla.github.io/statsforecast/src/adapters.prophet.html#autoarimaprophet)
adapter without external regressors. It inherits the `Prophet`
constructor as well as its `fit` and `predict` methods.

With the class
[`AutoARIMAProphet`](https://Nixtla.github.io/statsforecast/src/adapters.prophet.html#autoarimaprophet)
you can simply substitute `Prophet` and you’ll be training an
[`AutoARIMA`](https://Nixtla.github.io/statsforecast/src/core/models.html#autoarima)
model without changing anything in your forecasting pipeline.

```python
m = AutoARIMAProphet(daily_seasonality=False)
m.fit(df)
# m.fit(df, disable_seasonal_features=False) # Uncomment for better AutoARIMA predictions
future = m.make_future_dataframe(365)
forecast = m.predict(future)
```

```python
fig = m.plot(forecast)
```

## 2.2 Holiday Prophet

Usually `Prophet` pipelines include the usage of external regressors
such as **holidays**.

Suppose you want to include holidays or other recurring calendar events,
you can create a pandas.DataFrame for them. The DataFrame needs two
columns \[`holiday`, `ds`\] and a row for each holiday. It requires all
the occurrences of the holiday (as far as the historical data allows)
and the future events of the holiday. If the future does not have the
holidays registered, they will be modeled but not included in the
forecast.

You can also include into the events DataFrame, `lower_window` and
`upper_window` that extends the effect of the holidays through dates to
\[`lower_window`, `upper_window`\] days around the date. For example if
you wanted to account for Christmas Eve in addition to Christmas you’d
include `lower_window=-1`,`upper_window=0`, or Black Friday in addition
to Thanksgiving, you’d include `lower_window=0`,`upper_window=1`.

Here we Peyton Manning’s playoff appearances dates:

```python
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
```

```python
m = Prophet(daily_seasonality=False, holidays=holidays)
m.add_country_holidays(country_name='US')
m.fit(df)
future = m.make_future_dataframe(365)
forecast = m.predict(future)
```

```python
fig = m.plot(forecast)
```

The class
[`AutoARIMAProphet`](https://Nixtla.github.io/statsforecast/src/adapters.prophet.html#autoarimaprophet)
adapter allows to handle these scenarios to fit an
[`AutoARIMA`](https://Nixtla.github.io/statsforecast/src/core/models.html#autoarima)
model with exogenous variables.

You can enjoy your Prophet pipelines with the improved performance of a
classic ARIMA.

```python
m = AutoARIMAProphet(daily_seasonality=False,
                     holidays=holidays)
m.add_country_holidays(country_name='US')
m.fit(df)
# m.fit(df, disable_seasonal_features=False) # Uncomment for better AutoARIMA predictions
future = m.make_future_dataframe(365)
forecast = m.predict(future)
```

```python
fig = m.plot(forecast)
```
