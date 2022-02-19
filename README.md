# Statistical ⚡️ Forecast
> Lightning fast forecasting with statistical and econometric models


<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/branding/logo_mid.png">

[![CI](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/statsforecast)](https://pypi.org/project/statsforecast/)
[![PyPi](https://img.shields.io/pypi/v/statsforecast?color=blue)](https://pypi.org/project/statsforecast/)
[![License](https://img.shields.io/github/license/Nixtla/statsforecast)](https://github.com/Nixtla/statsforecast/blob/main/LICENSE)


**statsforecast** offers a collection of widly used univariate time series forecasting models including exponential smoothing and automatic ARIMA modelling optimized for high perfomance using `numba`.

## 🔥 Features

* Fastest and most acurate autoarima in Python and R (for the moment...)
* Out of the box implementation of other classical models and benchmarks like exponetial smoothing, croston, sesonal naive, random walk and tbs.
* 20x faster than pmdarima
* 1.5x faster than R
* 500x faster than Prophet 
* Compiled to high performance machine code through [numba](https://numba.pydata.org/)

## 📖 Why? 

Current python alternatives for statistical models are slow and innacurate. So we created a librabry that can be used to forecast in production enviorments or as benchmarks.  `StatsForecast` includes a large battery of models that can efficiently fit thousands of time series in an efficient manner. 

### 🔬 Accuracy

We compared accuracy and speed of this library against: [`pmdarima`](https://github.com/alkaline-ml/pmdarima), Bob Hyndman's [forecast](https://github.com/robjhyndman/forecast) package and Facebook's [prohpet](https://github.com/facebook/prophet). We used the `Daily`, `Hourly` and `Weekly` data from the [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128). 

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

### ⏲ Computational efficiency

Data scientists and developers have to iterate their models quickly in order to select the best approach and, once selected, they need a fast solution to deploy it into production so that business decisions can be made in a reasonable amount of time. Therefore, we compared our implementation in computational time based on the number of time series. The following graph shows the results. As we can see, the best model is our `auto_arima`. According to the table above, the computational performance does not compromise the accuracy.

![](nbs/imgs/computational-efficiency.png)

You can reproduce the results [here](/experiments/arima/).

## 💻 Install
`pip install statsforecast`

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

## 🔨 How to contribute
See [CONTRIBUTING.md](https://github.com/Nixtla/neuralforecast/blob/main/CONTRIBUTING.md).

## 📃 References
*  The `auto_arima` model is based (translated) from the R implementation included in the [forecast](https://github.com/robjhyndman/forecast) package developed by Rob Hyndman.
