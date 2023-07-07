---
title: Automatic Time Series Forecasting
---

> How to do automatic forecasting using `AutoARIMA`, `AutoETS`,
> `AutoCES` and `AutoTheta`.

:::tip

Automatic forecasts of large numbers of univariate time series are often
needed. It is common to have multiple product lines or skus that need
forecasting. In these circumstances, an automatic forecasting algorithm
is an essential tool. Automatic forecasting algorithms must determine an
appropriate time series model, estimate the parameters and compute the
forecasts. They must be robust to unusual time series patterns, and
applicable to large numbers of series without user intervention.

:::

## 1. Install statsforecast and load data {#install-statsforecast-and-load-data}

Use pip to install statsforecast and load Air Passangers dataset as an
example

<details>
<summary>Code</summary>

``` python
%%capture
!pip install statsforecast

from statsforecast.utils import AirPassengersDF

Y_df = AirPassengersDF
```

</details>

## 2. Import StatsForecast and models {#import-statsforecast-and-models}

Import the core StatsForecast class and the models you want to use

<details>
<summary>Code</summary>

``` python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, AutoCES
```

</details>

## 3. Instatiate the class {#instatiate-the-class}

Instantiate the StatsForecast class with the appropriate parameters

<details>
<summary>Code</summary>

``` python
season_length = 12 # Define season length as 12 months for monthly data
horizon = 1 # Forecast horizon is set to 1 month

# Define a list of models for forecasting
models = [
    AutoARIMA(season_length=season_length), # ARIMA model with automatic order selection and seasonal component
    AutoETS(season_length=season_length), # ETS model with automatic error, trend, and seasonal component
    AutoTheta(season_length=season_length), # Theta model with automatic seasonality detection
    AutoCES(season_length=season_length), # CES model with automatic seasonality detection
]

# Instantiate StatsForecast class with models, data frequency ('M' for monthly),
# and parallel computation on all CPU cores (n_jobs=-1)
sf = StatsForecast(
    models=models, # models for forecasting
    freq='M',  # frequency of the data
    n_jobs=-1  # number of jobs to run in parallel, -1 means using all processors
)
```

</details>

## 4. a) Forecast with forecast method {#a-forecast-with-forecast-method}

The `.forecast` method is faster for distributed computing and does not
save the fittted models

<details>
<summary>Code</summary>

``` python
# Generate forecasts for the specified horizon using the sf object
Y_hat_df = sf.forecast(df=Y_df, horizon=horizon) # forecast data

# Display the first few rows of the forecast DataFrame
Y_hat_df.head() # preview of forecasted data
```

</details>

## 4. b) Forecast with fit and predict {#b-forecast-with-fit-and-predict}

The `.fit` method saves the fitted models

<details>
<summary>Code</summary>

``` python
sf.fit(df=Y_df) # Fit the models to the data using the fit method of the StatsForecast object

sf.fitted_ # Access fitted models from the StatsForecast object

Y_hat_df = sf.predict(h=horizon) # Predict or forecast 'horizon' steps ahead using the predict method

Y_hat_df.head() # Preview the first few rows of the forecasted data
```

</details>

## References {#references}

[Hyndman, RJ and Khandakar, Y (2008) “Automatic time series forecasting:
The forecast package for R”, Journal of Statistical Software,
26(3).](https://www.jstatsoft.org/article/view/v027i03)

