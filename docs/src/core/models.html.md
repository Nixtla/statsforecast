# Models

StatsForecast offers a wide variety of statistical forecasting models grouped into the following categories:

- **Auto Forecast**: Automatic forecasting tools that search for the best parameters and select the best possible model. Useful for large collections of univariate time series. Includes: AutoARIMA, AutoETS, AutoTheta, AutoCES, AutoMFLES, AutoTBATS.

- **ARIMA Family**: AutoRegressive Integrated Moving Average models for capturing autocorrelations in time series data.

- **Exponential Smoothing**: Uses weighted averages of past observations where weights decrease exponentially into the past. Suitable for data with clear trend and/or seasonality.

- **Baseline Models**: Classical models for establishing baselines: HistoricAverage, Naive, RandomWalkWithDrift, SeasonalNaive, WindowAverage, SeasonalWindowAverage.

- **Sparse or Intermittent**: Models suited for series with very few non-zero observations: ADIDA, CrostonClassic, CrostonOptimized, CrostonSBA, IMAPA, TSB.

- **Multiple Seasonalities**: Models suited for signals with more than one clear seasonality. Useful for low-frequency data like electricity and logs: MSTL, MFLES, TBATS.

- **Theta Models**: Fit two theta lines to a deseasonalized time series using different techniques: Theta, OptimizedTheta, DynamicTheta, DynamicOptimizedTheta.

- **ARCH/GARCH Family**: Models for time series exhibiting non-constant volatility over time. Commonly used in finance.

- **Machine Learning**: Wrapper for scikit-learn models to be used with StatsForecast.

## Automatic Forecasting

### AutoARIMA

::: statsforecast.models.AutoARIMA
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### AutoETS

::: statsforecast.models.AutoETS
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### AutoCES

::: statsforecast.models.AutoCES
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### AutoTheta

::: statsforecast.models.AutoTheta
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### AutoMFLES

::: statsforecast.models.AutoMFLES
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### AutoTBATS

::: statsforecast.models.AutoTBATS
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

## ARIMA Family

### ARIMA

::: statsforecast.models.ARIMA
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### AutoRegressive

::: statsforecast.models.AutoRegressive
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

## Exponential Smoothing

### SimpleExponentialSmoothing

::: statsforecast.models.SimpleExponentialSmoothing
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### SimpleExponentialSmoothingOptimized

::: statsforecast.models.SimpleExponentialSmoothingOptimized
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### SeasonalExponentialSmoothing

::: statsforecast.models.SeasonalExponentialSmoothing
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### SeasonalExponentialSmoothingOptimized

::: statsforecast.models.SeasonalExponentialSmoothingOptimized
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### Holt

::: statsforecast.models.Holt
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

### HoltWinters

::: statsforecast.models.HoltWinters
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

## Baseline Models

### HistoricAverage

::: statsforecast.models.HistoricAverage
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### Naive

::: statsforecast.models.Naive
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### RandomWalkWithDrift

::: statsforecast.models.RandomWalkWithDrift
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### SeasonalNaive

::: statsforecast.models.SeasonalNaive
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### WindowAverage

::: statsforecast.models.WindowAverage
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### SeasonalWindowAverage

::: statsforecast.models.SeasonalWindowAverage
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

## Sparse or Intermittent Models

### ADIDA

::: statsforecast.models.ADIDA
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### CrostonClassic

::: statsforecast.models.CrostonClassic
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### CrostonOptimized

::: statsforecast.models.CrostonOptimized
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### CrostonSBA

::: statsforecast.models.CrostonSBA
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### IMAPA

::: statsforecast.models.IMAPA
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### TSB

::: statsforecast.models.TSB
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

## Multiple Seasonalities

### MSTL

::: statsforecast.models.MSTL
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### MFLES

::: statsforecast.models.MFLES
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### TBATS

::: statsforecast.models.TBATS
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

## Theta Family

### Theta

::: statsforecast.models.Theta
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

### OptimizedTheta

::: statsforecast.models.OptimizedTheta
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

### DynamicTheta

::: statsforecast.models.DynamicTheta
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

### DynamicOptimizedTheta

::: statsforecast.models.DynamicOptimizedTheta
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

## ARCH/GARCH Family

### GARCH

::: statsforecast.models.GARCH
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### ARCH

::: statsforecast.models.ARCH
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

## Machine Learning

### SklearnModel

::: statsforecast.models.SklearnModel
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

## Fallback Models

These models are used as fallbacks when other models fail during forecasting.

### ConstantModel

::: statsforecast.models.ConstantModel
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__
        - fit
        - predict
        - predict_in_sample
        - forecast

### ZeroModel

::: statsforecast.models.ZeroModel
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

### NaNModel

::: statsforecast.models.NaNModel
    options:
      show_source: true
      heading_level: 4
      members:
        - __init__

## Usage Examples

### Basic Model Usage

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive
from statsforecast.utils import generate_series

# Generate example data
df = generate_series(n_series=10)

# Create StatsForecast instance with models
sf = StatsForecast(
    models=[
        AutoARIMA(season_length=7),
        Naive()
    ],
    freq='D'
)

# Forecast
forecasts = sf.forecast(df=df, h=7)
```

### Using Multiple Models

```python
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    SeasonalNaive,
    Theta,
    HistoricAverage
)

# Combine multiple models for comparison
models = [
    AutoARIMA(season_length=12),
    AutoETS(season_length=12),
    SeasonalNaive(season_length=12),
    Theta(season_length=12),
    HistoricAverage()
]

sf = StatsForecast(models=models, freq='M', n_jobs=-1)
forecasts = sf.forecast(df=df, h=12, level=[80, 95])
```

### Model with Prediction Intervals

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import ConformalIntervals

# Create model with conformal prediction intervals
model = AutoARIMA(
    season_length=12,
    prediction_intervals=ConformalIntervals(n_windows=2, h=12),
    alias='ConformalAutoARIMA'
)

sf = StatsForecast(models=[model], freq='M')
forecasts = sf.forecast(df=df, h=12, level=[80, 95])
```

### Sparse/Intermittent Data

```python
from statsforecast import StatsForecast
from statsforecast.models import (
    CrostonOptimized,
    ADIDA,
    IMAPA,
    TSB
)

# Models specialized for sparse/intermittent data
sparse_models = [
    CrostonOptimized(),
    ADIDA(),
    IMAPA(),
    TSB(alpha_d=0.2, alpha_p=0.2)
]

sf = StatsForecast(models=sparse_models, freq='D')
forecasts = sf.forecast(df=sparse_df, h=30)
```

### Multiple Seasonalities

```python
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoTBATS

# For data with multiple seasonal patterns
models = [
    MSTL(season_length=[24, 168]),  # Hourly with daily and weekly seasonality
    AutoTBATS(season_length=[24, 168])
]

sf = StatsForecast(models=models, freq='H')
forecasts = sf.forecast(df=hourly_df, h=168)
```

### ARCH/GARCH for Volatility

```python
from statsforecast import StatsForecast
from statsforecast.models import GARCH, ARCH

# Models for financial data with volatility
volatility_models = [
    GARCH(p=1, q=1),
    ARCH(p=1)
]

sf = StatsForecast(models=volatility_models, freq='D')
forecasts = sf.forecast(df=financial_df, h=30)
```

### Using Scikit-learn Models

```python
from statsforecast import StatsForecast
from statsforecast.models import SklearnModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Wrap scikit-learn models
models = [
    SklearnModel(RandomForestRegressor(n_estimators=100), alias='RF'),
    SklearnModel(Ridge(alpha=1.0), alias='Ridge')
]

sf = StatsForecast(models=models, freq='D')
forecasts = sf.forecast(df=df, h=14)
```

## Model Selection Tips

- **For automatic selection**: Start with `AutoARIMA` or `AutoETS`
- **For baseline comparison**: Use `Naive`, `SeasonalNaive`, or `HistoricAverage`
- **For seasonal data**: Use models with `season_length` parameter
- **For sparse data**: Use Croston family or ADIDA
- **For multiple seasonalities**: Use MSTL or TBATS
- **For volatile data**: Use GARCH or ARCH
- **For ensemble approaches**: Combine multiple models and compare performance

## References

For detailed information on the statistical models and algorithms, please refer to the [source code](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/models.py) and the original academic papers referenced in the docstrings.
