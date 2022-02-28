# <center>Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)</center>
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->



<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/branding/logo_mid.png">
<h1 align="center">Statistical ⚡️ Forecast</h1>
<h3 align="center">Lightning fast forecasting with statistical and econometric models</h3>
    
[![CI](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/statsforecast)](https://pypi.org/project/statsforecast/)
[![PyPi](https://img.shields.io/pypi/v/statsforecast?color=blue)](https://pypi.org/project/statsforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/nixtla/statsforecast?color=seagreen&label=conda)](https://anaconda.org/nixtla/statsforecast)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/Nixtla/statsforecast/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/statsforecast.svg?label=docs)](https://nixtla.github.io/statsforecast/)  
    
**StatsForecast** offers a collection of widely used univariate time series forecasting models, including exponential smoothing and automatic `ARIMA` modeling optimized for high performance using `numba`.

[Getting started](#%F0%9F%A7%AC%20Getting%20Started) •
[Installation](#💻-installation)
</div>

## 🔥 Features

* Fastest and most accurate `auto_arima` in Python and R (for the moment...).
* Out of the box implementation of other classical models and benchmarks like `exponential smoothing`, `croston`, `sesonal naive`, `random walk with drift` and `tbs`.
* 20x faster than `pmdarima`.
* 1.5x faster than R.
* 500x faster than `Prophet`. 
* Compiled to high performance machine code through [`numba`](https://numba.pydata.org/).

## 📖 Why? 

Current Python alternatives for statistical models are slow and inaccurate. So we created a library that can be used to forecast in production environments or as benchmarks.  `StatsForecast` includes an extensive battery of models that can efficiently fit thousands of time series.

### 🔬 Accuracy

We compared accuracy and speed against: [pmdarima](https://github.com/alkaline-ml/pmdarima), Rob Hyndman's [forecast](https://github.com/robjhyndman/forecast) package and Facebook's [Prophet](https://github.com/facebook/prophet). We used the `Daily`, `Hourly` and `Weekly` data from the [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128). 

The following table summarizes the results. As can be seen, our `auto_arima` is the best model in accuracy (measured by the `MASE` loss) and time, even compared with the original implementation in R.

| dataset   | metric   |   nixtla | pmdarima [1] |   auto_arima_r |   prophet |
|:----------|:---------|--------------------:|----------------------:|---------------:|----------:|
| M4-Daily     | MASE     |                **3.26** |                  3.35 |           4.46 |     14.26 |
| M4-Daily     | time     |                **1.41** |                 27.61 |           1.81 |    514.33 |
| M4-Hourly    | MASE     |                **0.92** |                ---    |           1.02 |      1.78 |
| M4-Hourly    | time     |               **12.92** |                ---    |          23.95 |     17.27 |
| M4-Weekly    | MASE     |                **2.34** |                  2.47 |           2.58 |      7.29 |
| M4-Weekly    | time     |                0.42 |                  2.92 |           **0.22** |     19.82 |


[1] The model `auto_arima` from `pmdarima` had problems with Hourly data. An issue was opened in their repo.

The following table summarizes the data details.
 
| group   | n_series   | mean_length   | std_length   |   min_length | max_length   |
|:--------|-----------:|--------------:|-------------:|-------------:|-------------:|
| Daily   | 4,227      | 2,371         | 1,756        |          107 | 9,933        |
| Hourly  | 414        | 901           | 127          |          748 | 1,008        |
| Weekly  | 359        | 1,035         | 707          |           93 | 2,610        | 

### ⏲ Computational efficiency

We measured the computational time against the number of time series. The following graph shows the results. As we can see, the fastest model is our `auto_arima`.

![](nbs/imgs/computational-efficiency.png)

<details>
    <summary> Nixtla vs Prophet </summary> 
    <img src="nbs/imgs/computational-efficiency-hours-wo-pmdarima.png" > 
</details>

You can reproduce the results [here](/experiments/arima/).

## 👾 Less code
![pmd to stats](nbs/imgs/pdmarimaStats.gif)


## 📖 Documentation
Here is a link to the [documentation](https://nixtla.github.io/statsforecast/).

## 🧬 Getting Started [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/statsforecast/blob/main/examples/arima.ipynb)

[Example Jupyter Notebook](https://github.com/Nixtla/statsforecast/blob/main/examples/arima.ipynb)

## 💻 Installation
<details>
<summary>PyPI</summary>

You can install the *released version* of `StatsForecast` from the [Python package index](https://pypi.org) with:

```python
pip install statsforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
</details>

<details>
<summary>Conda</summary>
  
Also you can install the *released version* of `StatsForecast` from [conda](https://anaconda.org) with:

```python
conda install -c nixtla statsforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
</details>

<details>
<summary>Dev Mode</summary>
If you want to make some modifications to the code and see the effects in real time (without reinstalling), follow the steps below:

```bash
git clone https://github.com/Nixtla/statsforecast.git
cd statsforecast
pip install -e .
```
</details>

## 🧬 How to use

```python
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import seasonal_naive, auto_arima
from statsforecast.utils import AirPassengers
```

```python
horizon = 12
ap_train = AirPassengers[:-horizon]
ap_test = AirPassengers[-horizon:]
```

```python
series_train = pd.DataFrame(
    {
        'ds': np.arange(1, ap_train.size + 1),
        'y': ap_train
    },
    index=pd.Index([0] * ap_train.size, name='unique_id')
)
```

```python
def display_df(df):
    display(Markdown(df.to_markdown()))
```

```python
fcst = StatsForecast(
    series_train, 
    models=[(auto_arima, 12), (seasonal_naive, 12)], 
    freq='M', 
    n_jobs=1
)
forecasts = fcst.forecast(12)
display_df(forecasts)
```


|   unique_id |   ds |   auto_arima_season_length-12 |   seasonal_naive_season_length-12 |
|------------:|-----:|------------------------------:|----------------------------------:|
|           0 |  133 |                       424.16  |                               360 |
|           0 |  134 |                       407.082 |                               342 |
|           0 |  135 |                       470.861 |                               406 |
|           0 |  136 |                       460.914 |                               396 |
|           0 |  137 |                       484.901 |                               420 |
|           0 |  138 |                       536.904 |                               472 |
|           0 |  139 |                       612.903 |                               548 |
|           0 |  140 |                       623.903 |                               559 |
|           0 |  141 |                       527.903 |                               463 |
|           0 |  142 |                       471.903 |                               407 |
|           0 |  143 |                       426.903 |                               362 |
|           0 |  144 |                       469.903 |                               405 |


```python
forecasts['y_test'] = ap_test
```

```python
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
pd.concat([series_train, forecasts]).set_index('ds').plot(ax=ax, linewidth=2)
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
```


    
![png](docs/images/output_23_0.png)
    


### Adding external regressors

```python
series_xreg = pd.DataFrame(
    {
        'ds': pd.date_range(start='1949-01-01', periods=ap_train.size, freq='M'),
        'y': ap_train
    },
    index=pd.Index([0] * ap_train.size, name='unique_id')
)
```

```python
series_xreg['trend'] = np.arange(1, ap_train.size + 1)
series_xreg['intercept'] = np.ones(ap_train.size)
series_xreg['month'] = series_xreg['ds'].dt.month
series_xreg = pd.get_dummies(series_xreg, columns=['month'], drop_first=True)
```

```python
display_df(series_xreg.head())
```


|   unique_id | ds                  |   y |   trend |   intercept |   month_2 |   month_3 |   month_4 |   month_5 |   month_6 |   month_7 |   month_8 |   month_9 |   month_10 |   month_11 |   month_12 |
|------------:|:--------------------|----:|--------:|------------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|
|           0 | 1949-01-31 00:00:00 | 112 |       1 |           1 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |
|           0 | 1949-02-28 00:00:00 | 118 |       2 |           1 |         1 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |
|           0 | 1949-03-31 00:00:00 | 132 |       3 |           1 |         0 |         1 |         0 |         0 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |
|           0 | 1949-04-30 00:00:00 | 129 |       4 |           1 |         0 |         0 |         1 |         0 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |
|           0 | 1949-05-31 00:00:00 | 121 |       5 |           1 |         0 |         0 |         0 |         1 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |


```python
xreg_test = pd.DataFrame(
    {
        'ds': pd.date_range(start='1960-01-01', periods=ap_test.size, freq='M')
    },
    index=pd.Index([0] * ap_test.size, name='unique_id')
)
```

```python
xreg_test['trend'] = np.arange(133, ap_test.size + 133)
xreg_test['intercept'] = np.ones(ap_test.size)
xreg_test['month'] = xreg_test['ds'].dt.month
xreg_test = pd.get_dummies(xreg_test, columns=['month'], drop_first=True)
```

```python
fcst = StatsForecast(
    series_xreg, 
    models=[(auto_arima, 12), (seasonal_naive, 12)], 
    freq='M', 
    n_jobs=1
)
forecasts = fcst.forecast(12, xreg=xreg_test)
display_df(forecasts)
```


|   unique_id | ds                  |   auto_arima_season_length-12 |   seasonal_naive_season_length-12 |
|------------:|:--------------------|------------------------------:|----------------------------------:|
|           0 | 1960-01-31 00:00:00 |                       392.207 |                               360 |
|           0 | 1960-02-29 00:00:00 |                       376.962 |                               342 |
|           0 | 1960-03-31 00:00:00 |                       436.133 |                               406 |
|           0 | 1960-04-30 00:00:00 |                       427.066 |                               396 |
|           0 | 1960-05-31 00:00:00 |                       459.206 |                               420 |
|           0 | 1960-06-30 00:00:00 |                       505.109 |                               472 |
|           0 | 1960-07-31 00:00:00 |                       583.265 |                               548 |
|           0 | 1960-08-31 00:00:00 |                       596.063 |                               559 |
|           0 | 1960-09-30 00:00:00 |                       491.6   |                               463 |
|           0 | 1960-10-31 00:00:00 |                       429.576 |                               407 |
|           0 | 1960-11-30 00:00:00 |                       394.843 |                               362 |
|           0 | 1960-12-31 00:00:00 |                       433.5   |                               405 |


```python
forecasts['y_test'] = ap_test
```

```python
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
pd.concat([series_xreg[['ds', 'y']], forecasts]).set_index('ds').plot(ax=ax, linewidth=2)
ax.set_title('AirPassengers Forecast using External Regressors', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
```


    
![png](docs/images/output_32_0.png)
    


## 🔨 How to contribute
See [CONTRIBUTING.md](https://github.com/Nixtla/neuralforecast/blob/main/CONTRIBUTING.md).

## 📃 References
*  The `auto_arima` model is based (translated) from the R implementation included in the [forecast](https://github.com/robjhyndman/forecast) package developed by Rob Hyndman.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/FedericoGarza"><img src="https://avatars.githubusercontent.com/u/10517170?v=4?s=100" width="100px;" alt=""/><br /><sub><b>fede</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=FedericoGarza" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!