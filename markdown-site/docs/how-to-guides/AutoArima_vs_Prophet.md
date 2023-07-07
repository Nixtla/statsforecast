---
title: AutoARIMA Comparison (Prophet and pmdarima)
---

export const quartoRawHtml =
[`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>`,`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>`,`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
<p>768 rows × 3 columns</p>
</div>`,`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>`,`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>`,`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>`];

<a href="https://colab.research.google.com/github/Nixtla/statsforecast/blob/main/nbs/examples/AutoArima_vs_Prophet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Motivation {#motivation}

The `AutoARIMA` model is widely used to forecast time series in
production and as a benchmark. However, the python implementation
(`pmdarima`) is so slow that prevent data scientist practioners from
quickly iterating and deploying `AutoARIMA` in production for a large
number of time series. In this notebook we present Nixtla’s `AutoARIMA`
based on the R implementation (developed by Rob Hyndman) and optimized
using `numba`.

## Example {#example}

### Libraries {#libraries}

<details>
<summary>Code</summary>

``` python
%%capture
!pip install statsforecast prophet statsmodels sklearn matplotlib pmdarima
```

</details>
<details>
<summary>Code</summary>

``` python
import logging
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")
from itertools import product
from multiprocessing import cpu_count, Pool # for prophet

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima as auto_arima_p
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, _TS
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import ParameterGrid
```

</details>

``` text
Importing plotly failed. Interactive plots will not work.
```

#### Useful functions {#useful-functions}

The `plot_grid` function defined below will be useful to plot different
time series, and different models’ forecasts.

<details>
<summary>Code</summary>

``` python
def plot_grid(df_train, df_test=None, plot_random=True):
    fig, axes = plt.subplots(4, 2, figsize = (24, 14))

    unique_ids = df_train['unique_id'].unique()

    assert len(unique_ids) >= 8, "Must provide at least 8 ts"
    
    if plot_random:
        unique_ids = random.sample(list(unique_ids), k=8)
    else:
        unique_uids = unique_ids[:8]

    for uid, (idx, idy) in zip(unique_ids, product(range(4), range(2))):
        train_uid = df_train.query('unique_id == @uid')
        axes[idx, idy].plot(train_uid['ds'], train_uid['y'], label = 'y_train')
        if df_test is not None:
            max_ds = train_uid['ds'].max()
            test_uid = df_test.query('unique_id == @uid')
            for model in df_test.drop(['unique_id', 'ds'], axis=1).columns:
                if all(np.isnan(test_uid[model])):
                    continue
                axes[idx, idy].plot(test_uid['ds'], test_uid[model], label=model)

        axes[idx, idy].set_title(f'M4 Hourly: {uid}')
        axes[idx, idy].set_xlabel('Timestamp [t]')
        axes[idx, idy].set_ylabel('Target')
        axes[idx, idy].legend(loc='upper left')
        axes[idx, idy].xaxis.set_major_locator(plt.MaxNLocator(20))
        axes[idx, idy].grid()
    fig.subplots_adjust(hspace=0.5)
    plt.show()
```

</details>
<details>
<summary>Code</summary>

``` python
def plot_autocorrelation_grid(df_train):
    fig, axes = plt.subplots(4, 2, figsize = (24, 14))

    unique_ids = df_train['unique_id'].unique()

    assert len(unique_ids) >= 8, "Must provide at least 8 ts"

    unique_ids = random.sample(list(unique_ids), k=8)

    for uid, (idx, idy) in zip(unique_ids, product(range(4), range(2))):
        train_uid = df_train.query('unique_id == @uid')
        plot_acf(train_uid['y'].values, ax=axes[idx, idy], 
                 title=f'ACF M4 Hourly {uid}')
        axes[idx, idy].set_xlabel('Timestamp [t]')
        axes[idx, idy].set_ylabel('Autocorrelation')
    fig.subplots_adjust(hspace=0.5)
    plt.show()
```

</details>

### Data {#data}

For testing purposes, we will use the Hourly dataset from the M4
competition.

<details>
<summary>Code</summary>

``` python
%%capture
!wget https://auto-arima-results.s3.amazonaws.com/M4-Hourly.csv
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
!wget https://auto-arima-results.s3.amazonaws.com/M4-Hourly-test.csv
```

</details>
<details>
<summary>Code</summary>

``` python
train = pd.read_csv('M4-Hourly.csv')
test = pd.read_csv('M4-Hourly-test.csv').rename(columns={'y': 'y_test'})
```

</details>

In this example we will use a subset of the data to avoid waiting too
long. You can modify the number of series if you want.

<details>
<summary>Code</summary>

``` python
n_series = 16
uids = train['unique_id'].unique()[:n_series]
train = train.query('unique_id in @uids')
test = test.query('unique_id in @uids')
```

</details>
<details>
<summary>Code</summary>

``` python
plot_grid(train, test)
```

</details>

![](AutoArima_vs_Prophet_files/figure-markdown_strict/cell-10-output-1.png)

Would an autorregresive model be the right choice for our data? There is
no doubt that we observe seasonal periods. The autocorrelation function
(`acf`) can help us to answer the question. Intuitively, we have to
observe a decreasing correlation to opt for an AR model.

<details>
<summary>Code</summary>

``` python
plot_autocorrelation_grid(train)
```

</details>

![](AutoArima_vs_Prophet_files/figure-markdown_strict/cell-11-output-1.png)

Thus, we observe a high autocorrelation for previous lags and also for
the seasonal lags. Therefore, we will let `auto_arima` to handle our
data.

### Training and forecasting {#training-and-forecasting}

`StatsForecast` receives a list of models to fit each time series. Since
we are dealing with Hourly data, it would be benefitial to use 24 as
seasonality.

<details>
<summary>Code</summary>

``` python
?AutoARIMA
```

</details>

``` text
Init signature:
AutoARIMA(
    d: Optional[int] = None,
    D: Optional[int] = None,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_order: int = 5,
    max_d: int = 2,
    max_D: int = 1,
    start_p: int = 2,
    start_q: int = 2,
    start_P: int = 1,
    start_Q: int = 1,
    stationary: bool = False,
    seasonal: bool = True,
    ic: str = 'aicc',
    stepwise: bool = True,
    nmodels: int = 94,
    trace: bool = False,
    approximation: Optional[bool] = False,
    method: Optional[str] = None,
    truncate: Optional[bool] = None,
    test: str = 'kpss',
    test_kwargs: Optional[str] = None,
    seasonal_test: str = 'seas',
    seasonal_test_kwargs: Optional[Dict] = None,
    allowdrift: bool = False,
    allowmean: bool = False,
    blambda: Optional[float] = None,
    biasadj: bool = False,
    parallel: bool = False,
    num_cores: int = 2,
    season_length: int = 1,
)
Docstring:      <no docstring>
File:           ~/fede/statsforecast/statsforecast/models.py
Type:           type
Subclasses:     
```

As we see, we can pass `season_length` to `AutoARIMA`, so the definition
of our models would be,

<details>
<summary>Code</summary>

``` python
models = [AutoARIMA(season_length=24, approximation=True)]
```

</details>
<details>
<summary>Code</summary>

``` python
fcst = StatsForecast(df=train, 
                     models=models, 
                     freq='H', 
                     n_jobs=-1)
```

</details>
<details>
<summary>Code</summary>

``` python
init = time.time()
forecasts = fcst.forecast(48)
end = time.time()

time_nixtla = end - init
time_nixtla
```

</details>

``` text
20.36360502243042
```

<details>
<summary>Code</summary>

``` python
forecasts.head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

|           | ds  | AutoARIMA  |
|-----------|-----|------------|
| unique_id |     |            |
| H1        | 701 | 616.084167 |
| H1        | 702 | 544.432129 |
| H1        | 703 | 510.414490 |
| H1        | 704 | 481.046539 |
| H1        | 705 | 460.893066 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

<details>
<summary>Code</summary>

``` python
forecasts = forecasts.reset_index()
```

</details>
<details>
<summary>Code</summary>

``` python
test = test.merge(forecasts, how='left', on=['unique_id', 'ds'])
```

</details>
<details>
<summary>Code</summary>

``` python
plot_grid(train, test)
```

</details>

![](AutoArima_vs_Prophet_files/figure-markdown_strict/cell-19-output-1.png)

## Alternatives {#alternatives}

### pmdarima {#pmdarima}

You can use the `StatsForecast` class to parallelize your own models. In
this section we will use it to run the `auto_arima` model from
`pmdarima`.

<details>
<summary>Code</summary>

``` python
class PMDAutoARIMA(_TS):
    
    def __init__(self, season_length: int):
        self.season_length = season_length
        
    def forecast(self, y, h, X=None, X_future=None, fitted=False):
        mod = auto_arima_p(
            y, m=self.season_length,
            with_intercept=False #ensure comparability with Nixtla's implementation
        ) 
        return {'mean': mod.predict(h)}
    
    def __repr__(self):
        return 'pmdarima'
```

</details>
<details>
<summary>Code</summary>

``` python
n_series_pmdarima = 2
```

</details>
<details>
<summary>Code</summary>

``` python
fcst = StatsForecast(
    df = train.query('unique_id in ["H1", "H10"]'), 
    models=[PMDAutoARIMA(season_length=24)],
    freq='H',
    n_jobs=-1
)
```

</details>
<details>
<summary>Code</summary>

``` python
init = time.time()
forecast_pmdarima = fcst.forecast(48)
end = time.time()

time_pmdarima = end - init
time_pmdarima
```

</details>

``` text
349.93623208999634
```

<details>
<summary>Code</summary>

``` python
forecast_pmdarima.head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

|           | ds  | pmdarima   |
|-----------|-----|------------|
| unique_id |     |            |
| H1        | 701 | 627.479370 |
| H1        | 702 | 570.364380 |
| H1        | 703 | 541.831482 |
| H1        | 704 | 516.475647 |
| H1        | 705 | 503.044586 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

<details>
<summary>Code</summary>

``` python
forecast_pmdarima = forecast_pmdarima.reset_index()
```

</details>
<details>
<summary>Code</summary>

``` python
test = test.merge(forecast_pmdarima, how='left', on=['unique_id', 'ds'])
```

</details>
<details>
<summary>Code</summary>

``` python
plot_grid(train, test, plot_random=False)
```

</details>

![](AutoArima_vs_Prophet_files/figure-markdown_strict/cell-27-output-1.png)

### Prophet {#prophet}

`Prophet` is designed to receive a pandas dataframe, so we cannot use
`StatForecast`. Therefore, we need to parallize from scratch.

<details>
<summary>Code</summary>

``` python
params_grid = {'seasonality_mode': ['multiplicative','additive'],
               'growth': ['linear', 'flat'], 
               'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5], 
               'n_changepoints': [5, 10, 15, 20]} 
grid = ParameterGrid(params_grid)
```

</details>
<details>
<summary>Code</summary>

``` python
def fit_and_predict(index, ts):
    df = ts.drop(columns='unique_id', axis=1)
    max_ds = df['ds'].max()
    df['ds'] = pd.date_range(start='1970-01-01', periods=df.shape[0], freq='H')
    df_val = df.tail(48) 
    df_train = df.drop(df_val.index) 
    y_val = df_val['y'].values
    
    if len(df_train) >= 48:
        val_results = {'losses': [], 'params': []}

        for params in grid:
            model = Prophet(seasonality_mode=params['seasonality_mode'],
                            growth=params['growth'],
                            weekly_seasonality=True,
                            daily_seasonality=True,
                            yearly_seasonality=True,
                            n_changepoints=params['n_changepoints'],
                            changepoint_prior_scale=params['changepoint_prior_scale'])
            model = model.fit(df_train)
            
            forecast = model.make_future_dataframe(periods=48, 
                                                   include_history=False, 
                                                   freq='H')
            forecast = model.predict(forecast)
            forecast['unique_id'] = index
            forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])
            
            loss = np.mean(abs(y_val - forecast['yhat'].values))
            
            val_results['losses'].append(loss)
            val_results['params'].append(params)

        idx_params = np.argmin(val_results['losses']) 
        params = val_results['params'][idx_params]
    else:
        params = {'seasonality_mode': 'multiplicative',
                  'growth': 'flat',
                  'n_changepoints': 150,
                  'changepoint_prior_scale': 0.5}
    model = Prophet(seasonality_mode=params['seasonality_mode'],
                    growth=params['growth'],
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    yearly_seasonality=True,
                    n_changepoints=params['n_changepoints'],
                    changepoint_prior_scale=params['changepoint_prior_scale'])
    model = model.fit(df)
    
    forecast = model.make_future_dataframe(periods=48, 
                                           include_history=False, 
                                           freq='H')
    forecast = model.predict(forecast)
    forecast.insert(0, 'unique_id', index)
    forecast['ds'] = np.arange(max_ds + 1, max_ds + 48 + 1)
    forecast = forecast.filter(items=['unique_id', 'ds', 'yhat'])
    
    return forecast
```

</details>
<details>
<summary>Code</summary>

``` python
logging.getLogger('prophet').setLevel(logging.WARNING) 
```

</details>
<details>
<summary>Code</summary>

``` python
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
```

</details>
<details>
<summary>Code</summary>

``` python
init = time.time()
with suppress_stdout_stderr():
    with Pool(cpu_count()) as pool:
        forecast_prophet = pool.starmap(fit_and_predict, train.groupby('unique_id'))
end = time.time()
forecast_prophet = pd.concat(forecast_prophet).rename(columns={'yhat': 'prophet'})
time_prophet = end - init
time_prophet
```

</details>

``` text
2022-08-19 23:07:24 prophet.models WARNING: Optimization terminated abnormally. Falling back to Newton.
2022-08-19 23:07:25 prophet.models WARNING: Optimization terminated abnormally. Falling back to Newton.
2022-08-19 23:07:41 prophet.models WARNING: Optimization terminated abnormally. Falling back to Newton.
2022-08-19 23:07:42 prophet.models WARNING: Optimization terminated abnormally. Falling back to Newton.
2022-08-19 23:08:00 prophet.models WARNING: Optimization terminated abnormally. Falling back to Newton.
```

``` text
120.9244737625122
```

<details>
<summary>Code</summary>

``` python
forecast_prophet
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

|     | unique_id | ds  | prophet     |
|-----|-----------|-----|-------------|
| 0   | H1        | 701 | 631.867439  |
| 1   | H1        | 702 | 561.001661  |
| 2   | H1        | 703 | 499.299334  |
| 3   | H1        | 704 | 456.132082  |
| 4   | H1        | 705 | 431.884528  |
| ... | ...       | ... | ...         |
| 43  | H112      | 744 | 5634.503804 |
| 44  | H112      | 745 | 5622.643542 |
| 45  | H112      | 746 | 5546.302705 |
| 46  | H112      | 747 | 5457.777165 |
| 47  | H112      | 748 | 5373.944098 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

<details>
<summary>Code</summary>

``` python
test = test.merge(forecast_prophet, how='left', on=['unique_id', 'ds'])
```

</details>
<details>
<summary>Code</summary>

``` python
plot_grid(train, test)
```

</details>

![](AutoArima_vs_Prophet_files/figure-markdown_strict/cell-35-output-1.png)

### Evaluation {#evaluation}

### Time {#time}

Since `AutoARIMA` works with numba is useful to calculate the time for
just one time series.

<details>
<summary>Code</summary>

``` python
fcst = StatsForecast(df=train.query('unique_id == "H1"'), 
                     models=models, freq='H', 
                     n_jobs=1)
```

</details>
<details>
<summary>Code</summary>

``` python
init = time.time()
forecasts = fcst.forecast(48)
end = time.time()

time_nixtla_1 = end - init
time_nixtla_1 
```

</details>

``` text
11.437001705169678
```

<details>
<summary>Code</summary>

``` python
times = pd.DataFrame({'n_series': np.arange(1, 414 + 1)})
times['pmdarima'] = time_pmdarima * times['n_series'] / n_series_pmdarima
times['prophet'] = time_prophet * times['n_series'] / n_series
times['AutoARIMA_nixtla'] = time_nixtla_1 + times['n_series'] * (time_nixtla - time_nixtla_1) / n_series
times = times.set_index('n_series')
```

</details>
<details>
<summary>Code</summary>

``` python
times.tail(5)
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[6] }} />

|          | pmdarima     | prophet     | AutoARIMA_nixtla |
|----------|--------------|-------------|------------------|
| n_series |              |             |                  |
| 410      | 71736.927578 | 3098.689640 | 240.181212       |
| 411      | 71911.895694 | 3106.247420 | 240.739124       |
| 412      | 72086.863811 | 3113.805199 | 241.297037       |
| 413      | 72261.831927 | 3121.362979 | 241.854950       |
| 414      | 72436.800043 | 3128.920759 | 242.412863       |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[7] }} />

<details>
<summary>Code</summary>

``` python
fig, axes = plt.subplots(1, 2, figsize = (24, 7))
(times/3600).plot(ax=axes[0], linewidth=4)
np.log10(times).plot(ax=axes[1], linewidth=4)
axes[0].set_title('Time across models [Hours]', fontsize=22)
axes[1].set_title('Time across models [Log10 Scale]', fontsize=22)
axes[0].set_ylabel('Time [Hours]', fontsize=20)
axes[1].set_ylabel('Time Seconds [Log10 Scale]', fontsize=20)
fig.suptitle('Time comparison using M4-Hourly data', fontsize=27)
for ax in axes:
    ax.set_xlabel('Number of Time Series [N]', fontsize=20)
    ax.legend(prop={'size': 20})
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
```

</details>

![](AutoArima_vs_Prophet_files/figure-markdown_strict/cell-40-output-1.png)

<details>
<summary>Code</summary>

``` python
fig.savefig('computational-efficiency.png', dpi=300)
```

</details>

### Performance {#performance}

#### pmdarima (only two time series) {#pmdarima-only-two-time-series}

<details>
<summary>Code</summary>

``` python
name_models = test.drop(['unique_id', 'ds', 'y_test'], 1).columns.tolist()
```

</details>
<details>
<summary>Code</summary>

``` python
test_pmdarima = test.query('unique_id in ["H1", "H10"]')
eval_pmdarima = []
for model in name_models:
    mae = np.mean(abs(test_pmdarima[model] - test_pmdarima['y_test']))
    eval_pmdarima.append({'model': model, 'mae': mae})
pd.DataFrame(eval_pmdarima).sort_values('mae')
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[8] }} />

|     | model     | mae       |
|-----|-----------|-----------|
| 0   | AutoARIMA | 20.289669 |
| 1   | pmdarima  | 26.461525 |
| 2   | prophet   | 43.155861 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[9] }} />

#### Prophet {#prophet-1}

<details>
<summary>Code</summary>

``` python
eval_prophet = []
for model in name_models:
    if 'pmdarima' in model:
        continue
    mae = np.mean(abs(test[model] - test['y_test']))
    eval_prophet.append({'model': model, 'mae': mae})
pd.DataFrame(eval_prophet).sort_values('mae')
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[10] }} />

|     | model     | mae         |
|-----|-----------|-------------|
| 0   | AutoARIMA | 680.202970  |
| 1   | prophet   | 1066.049049 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[11] }} />

For a complete comparison check the [complete
experiment](https://github.com/Nixtla/statsforecast/tree/v0.6.0/experiments/arima).

<a href="https://colab.research.google.com/github/Nixtla/statsforecast/blob/main/nbs/examples/AutoArima_vs_Prophet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

