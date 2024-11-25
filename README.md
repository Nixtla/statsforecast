# Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-32-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
<h1 align="center">Statistical ⚡️ Forecast</h1>
<h3 align="center">Lightning fast forecasting with statistical and econometric models</h3>
    
[![CI](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/statsforecast)](https://pypi.org/project/statsforecast/)
[![PyPi](https://img.shields.io/pypi/v/statsforecast?color=blue)](https://pypi.org/project/statsforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/conda-forge/statsforecast?color=seagreen&label=conda)](https://anaconda.org/conda-forge/statsforecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/statsforecast/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/statsforecast.svg?label=docs)](https://nixtla.github.io/statsforecast/)
[![Downloads](https://pepy.tech/badge/statsforecast)](https://pepy.tech/project/statsforecast)
    
**StatsForecast** offers a collection of widely used univariate time series forecasting models, including automatic `ARIMA`, `ETS`, `CES`, and `Theta` modeling optimized for high performance using `numba`. It also includes a large battery of benchmarking models.
</div>

## Installation

You can install `StatsForecast` with:

```python
pip install statsforecast
```

or 

```python
conda install -c conda-forge statsforecast
``` 


Vist our [Installation Guide](https://nixtla.github.io/statsforecast/docs/getting-started/installation.html) for further instructions.

## Quick Start

**Minimal Example**

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

df = AirPassengersDF
sf = StatsForecast(
    models = [AutoARIMA(season_length = 12)],
    freq = 'M'
)

sf.fit(df)
sf.predict(h=12, level=[95])
```

**Get Started with this [quick guide](https://nixtla.github.io/statsforecast/docs/getting-started/getting_started_short.html).**

**Follow this [end-to-end walkthrough](https://nixtla.github.io/statsforecast/docs/getting-started/getting_started_complete.html) for best practices.**

## Why? 

Current Python alternatives for statistical models are slow, inaccurate and don't scale well. So we created a library that can be used to forecast in production environments or as benchmarks.  `StatsForecast` includes an extensive battery of models that can efficiently fit millions of time series.

## Features

* Fastest and most accurate implementations of `AutoARIMA`, `AutoETS`, `AutoCES`, `MSTL` and `Theta` in Python. 
* Out-of-the-box compatibility with Spark, Dask, and Ray.
* Probabilistic Forecasting and Confidence Intervals.
* Support for exogenous Variables and static covariates.
* Anomaly Detection.
* Familiar sklearn syntax: `.fit` and `.predict`.

## Highlights

* Inclusion of `exogenous variables` and `prediction intervals` for ARIMA.
* 20x [faster](./experiments/arima/) than `pmdarima`.
* 1.5x faster than `R`.
* 500x faster than `Prophet`. 
* 4x [faster](./experiments/ets/) than `statsmodels`.
* Compiled to high performance machine code through [`numba`](https://numba.pydata.org/).
* 1,000,000 series in [30 min](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray) with [ray](https://github.com/ray-project/ray).
* Replace FB-Prophet in two lines of code and gain speed and accuracy. Check the experiments [here](https://github.com/Nixtla/statsforecast/tree/main/experiments/arima_prophet_adapter).
* Fit 10 benchmark models on **1,000,000** series in [under **5 min**](./experiments/benchmarks_at_scale/). 


Missing something? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## Examples and Guides

📚 [End to End Walkthrough](https://nixtla.github.io/statsforecast/docs/getting-started/getting_started_complete.html): Model training, evaluation and selection for multiple time series

🔎 [Anomaly Detection](https://nixtla.github.io/statsforecast/docs/tutorials/anomalydetection.html): detect anomalies for time series using in-sample prediction intervals.

👩‍🔬 [Cross Validation](https://nixtla.github.io/statsforecast/docs/tutorials/crossvalidation.html): robust model’s performance evaluation.

❄️ [Multiple Seasonalities](https://nixtla.github.io/statsforecast/docs/tutorials/multipleseasonalities.html): how to forecast data with multiple seasonalities using an MSTL.

🔌 [Predict Demand Peaks](https://nixtla.github.io/statsforecast/docs/tutorials/electricitypeakforecasting.html): electricity load forecasting for detecting daily peaks and reducing electric bills.

📈 [Intermittent Demand](https://nixtla.github.io/statsforecast/docs/tutorials/intermittentdata.html): forecast series with very few non-zero observations. 

🌡️ [Exogenous Regressors](https://nixtla.github.io/statsforecast/docs/how-to-guides/exogenous.html): like weather or prices


## Models

### Automatic Forecasting
Automatic forecasting tools search for the best parameters and select the best possible model for a group of time series. These tools are useful for large collections of univariate time series.

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[AutoARIMA](https://nixtla.github.io/statsforecast/src/core/models.html#autoarima)|✅|✅|✅|✅|✅|
|[AutoETS](https://nixtla.github.io/statsforecast/src/core/models.html#autoets)|✅|✅|✅|✅||
|[AutoCES](https://nixtla.github.io/statsforecast/src/core/models.html#autoces)|✅|✅|✅|✅||
|[AutoTheta](https://nixtla.github.io/statsforecast/src/core/models.html#autotheta)|✅|✅|✅|✅||

### ARIMA Family
These models exploit the existing autocorrelations in the time series.

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[ARIMA](https://nixtla.github.io/statsforecast/src/core/models.html#arima)|✅|✅|✅|✅|✅|
|[AutoRegressive](https://nixtla.github.io/statsforecast/src/core/models.html#autoregressive)|✅|✅|✅|✅|✅|

### Theta Family
Fit two theta lines to a deseasonalized time series, using different techniques to obtain and combine the two theta lines to produce the final forecasts.

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[Theta](https://nixtla.github.io/statsforecast/src/core/models.html#theta)|✅|✅|✅|✅||
|[OptimizedTheta](https://nixtla.github.io/statsforecast/src/core/models.html#optimizedtheta)|✅|✅|✅|✅||
|[DynamicTheta](https://nixtla.github.io/statsforecast/src/core/models.html#dynamictheta)|✅|✅|✅|✅||
|[DynamicOptimizedTheta](https://nixtla.github.io/statsforecast/src/core/models.html#dynamicoptimizedtheta)|✅|✅|✅|✅||

### Multiple Seasonalities
Suited for signals with more than one clear seasonality. Useful for low-frequency data like electricity and logs.

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[MSTL](https://nixtla.github.io/statsforecast/src/core/models.html#mstl)|✅|✅|✅|✅|If trend forecaster supports|

### GARCH and ARCH Models 
Suited for modeling time series that exhibit non-constant volatility over time. The ARCH model is a particular case of GARCH. 

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[GARCH](https://nixtla.github.io/statsforecast/src/core/models.html#garch)|✅|✅|✅|✅||
|[ARCH](https://nixtla.github.io/statsforecast/src/core/models.html#arch)|✅|✅|✅|✅||


### Baseline Models
Classical models for establishing baseline.

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[HistoricAverage](https://nixtla.github.io/statsforecast/src/core/models.html#historicaverage)|✅|✅|✅|✅||
|[Naive](https://nixtla.github.io/statsforecast/src/core/models.html#naive)|✅|✅|✅|✅||
|[RandomWalkWithDrift](https://nixtla.github.io/statsforecast/src/core/models.html#randomwalkwithdrift)|✅|✅|✅|✅||
|[SeasonalNaive](https://nixtla.github.io/statsforecast/src/core/models.html#seasonalnaive)|✅|✅|✅|✅||
|[WindowAverage](https://nixtla.github.io/statsforecast/src/core/models.html#windowaverage)|✅|||||
|[SeasonalWindowAverage](https://nixtla.github.io/statsforecast/src/core/models.html#seasonalwindowaverage)|✅|||||

### Exponential Smoothing
Uses a weighted average of all past observations where the weights decrease exponentially into the past. Suitable for data with clear trend and/or seasonality. Use the `SimpleExponential` family for data with no clear trend or seasonality.

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[SimpleExponentialSmoothing](https://nixtla.github.io/statsforecast/src/core/models.html#simpleexponentialsmoothing)|✅|||||
|[SimpleExponentialSmoothingOptimized](https://nixtla.github.io/statsforecast/src/core/models.html#simpleexponentialsmoothingoptimized)|✅|||||
|[SeasonalExponentialSmoothing](https://nixtla.github.io/statsforecast/src/core/models.html#seasonalexponentialsmoothing)|✅|||||
|[SeasonalExponentialSmoothingOptimized](https://nixtla.github.io/statsforecast/src/core/models.html#seasonalexponentialsmoothingoptimized)|✅|||||
|[Holt](https://nixtla.github.io/statsforecast/src/core/models.html#holt)|✅|✅|✅|✅||
|[HoltWinters](https://nixtla.github.io/statsforecast/src/core/models.html#holtwinters)|✅|✅|✅|✅||


### Sparse or Intermittent 
Suited for series with very few non-zero observations

|Model | Point Forecast | Probabilistic Forecast | Insample fitted values | Probabilistic fitted values |Exogenous features|
|:------|:-------------:|:----------------------:|:---------------------:|:----------------------------:|:----------------:|
|[ADIDA](https://nixtla.github.io/statsforecast/src/core/models.html#adida)|✅||✅|✅||
|[CrostonClassic](https://nixtla.github.io/statsforecast/src/core/models.html#crostonclassic)|✅||✅|✅||
|[CrostonOptimized](https://nixtla.github.io/statsforecast/src/core/models.html#crostonoptimized)|✅||✅|✅||
|[CrostonSBA](https://nixtla.github.io/statsforecast/src/core/models.html#crostonsba)|✅||✅|✅||
|[IMAPA](https://nixtla.github.io/statsforecast/src/core/models.html#imapa)|✅||✅|✅||
|[TSB](https://nixtla.github.io/statsforecast/src/core/models.html#tsb)|✅||✅|✅||

## 🔨 How to contribute
See [CONTRIBUTING.md](https://github.com/Nixtla/statsforecast/blob/main/CONTRIBUTING.md).

## Citing

```bibtex
@misc{garza2022statsforecast,
    author={Azul Garza, Max Mergenthaler Canseco, Cristian Challú, Kin G. Olivares},
    title = {{StatsForecast}: Lightning fast forecasting with statistical and econometric models},
    year={2022},
    howpublished={{PyCon} Salt Lake City, Utah, US 2022},
    url={https://github.com/Nixtla/statsforecast}
}
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AzulGarza"><img src="https://avatars.githubusercontent.com/u/10517170?v=4?s=100" width="100px;" alt="azul"/><br /><sub><b>azul</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=AzulGarza" title="Code">💻</a> <a href="#maintenance-AzulGarza" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jmoralez"><img src="https://avatars.githubusercontent.com/u/8473587?v=4?s=100" width="100px;" alt="José Morales"/><br /><sub><b>José Morales</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=jmoralez" title="Code">💻</a> <a href="#maintenance-jmoralez" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/sugatoray/"><img src="https://avatars.githubusercontent.com/u/10201242?v=4?s=100" width="100px;" alt="Sugato Ray"/><br /><sub><b>Sugato Ray</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=sugatoray" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jefftackes.com"><img src="https://avatars.githubusercontent.com/u/9125316?v=4?s=100" width="100px;" alt="Jeff Tackes"/><br /><sub><b>Jeff Tackes</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Atackes" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/darinkist"><img src="https://avatars.githubusercontent.com/u/62692170?v=4?s=100" width="100px;" alt="darinkist"/><br /><sub><b>darinkist</b></sub></a><br /><a href="#ideas-darinkist" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alech97"><img src="https://avatars.githubusercontent.com/u/22159405?v=4?s=100" width="100px;" alt="Alec Helyar"/><br /><sub><b>Alec Helyar</b></sub></a><br /><a href="#question-alech97" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://dhirschfeld.github.io"><img src="https://avatars.githubusercontent.com/u/881019?v=4?s=100" width="100px;" alt="Dave Hirschfeld"/><br /><sub><b>Dave Hirschfeld</b></sub></a><br /><a href="#question-dhirschfeld" title="Answering Questions">💬</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mergenthaler"><img src="https://avatars.githubusercontent.com/u/4086186?v=4?s=100" width="100px;" alt="mergenthaler"/><br /><sub><b>mergenthaler</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=mergenthaler" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kdgutier"><img src="https://avatars.githubusercontent.com/u/19935241?v=4?s=100" width="100px;" alt="Kin"/><br /><sub><b>Kin</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=kdgutier" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Yasslight90"><img src="https://avatars.githubusercontent.com/u/58293883?v=4?s=100" width="100px;" alt="Yasslight90"/><br /><sub><b>Yasslight90</b></sub></a><br /><a href="#ideas-Yasslight90" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/asinig"><img src="https://avatars.githubusercontent.com/u/99350687?v=4?s=100" width="100px;" alt="asinig"/><br /><sub><b>asinig</b></sub></a><br /><a href="#ideas-asinig" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/guerda"><img src="https://avatars.githubusercontent.com/u/230782?v=4?s=100" width="100px;" alt="Philip Gillißen"/><br /><sub><b>Philip Gillißen</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=guerda" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shagn"><img src="https://avatars.githubusercontent.com/u/16029092?v=4?s=100" width="100px;" alt="Sebastian Hagn"/><br /><sub><b>Sebastian Hagn</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Ashagn" title="Bug reports">🐛</a> <a href="https://github.com/Nixtla/statsforecast/commits?author=shagn" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fugue-project/fugue"><img src="https://avatars.githubusercontent.com/u/21092479?v=4?s=100" width="100px;" alt="Han Wang"/><br /><sub><b>Han Wang</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=goodwanghan" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/benjamin-jeffrey-218548a8/"><img src="https://avatars.githubusercontent.com/u/36240394?v=4?s=100" width="100px;" alt="Ben Jeffrey"/><br /><sub><b>Ben Jeffrey</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Abjeffrey92" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Beliavsky"><img src="https://avatars.githubusercontent.com/u/38887928?v=4?s=100" width="100px;" alt="Beliavsky"/><br /><sub><b>Beliavsky</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=Beliavsky" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MMenchero"><img src="https://avatars.githubusercontent.com/u/47995617?v=4?s=100" width="100px;" alt="Mariana Menchero García "/><br /><sub><b>Mariana Menchero García </b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=MMenchero" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/guptanick/"><img src="https://avatars.githubusercontent.com/u/33585645?v=4?s=100" width="100px;" alt="Nikhil Gupta"/><br /><sub><b>Nikhil Gupta</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Angupta23" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jdegene"><img src="https://avatars.githubusercontent.com/u/17744939?v=4?s=100" width="100px;" alt="JD"/><br /><sub><b>JD</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Ajdegene" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jattenberg"><img src="https://avatars.githubusercontent.com/u/924185?v=4?s=100" width="100px;" alt="josh attenberg"/><br /><sub><b>josh attenberg</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=jattenberg" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JeroenPeterBos"><img src="https://avatars.githubusercontent.com/u/15342738?v=4?s=100" width="100px;" alt="JeroenPeterBos"/><br /><sub><b>JeroenPeterBos</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=JeroenPeterBos" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jvdd"><img src="https://avatars.githubusercontent.com/u/18898740?v=4?s=100" width="100px;" alt="Jeroen Van Der Donckt"/><br /><sub><b>Jeroen Van Der Donckt</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=jvdd" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Roymprog"><img src="https://avatars.githubusercontent.com/u/4035367?v=4?s=100" width="100px;" alt="Roymprog"/><br /><sub><b>Roymprog</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=Roymprog" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nelsoncardenas"><img src="https://avatars.githubusercontent.com/u/18086414?v=4?s=100" width="100px;" alt="Nelson Cárdenas Bolaño"/><br /><sub><b>Nelson Cárdenas Bolaño</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=nelsoncardenas" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kschmaus"><img src="https://avatars.githubusercontent.com/u/6586847?v=4?s=100" width="100px;" alt="Kyle Schmaus"/><br /><sub><b>Kyle Schmaus</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=kschmaus" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/akmal-soliev/"><img src="https://avatars.githubusercontent.com/u/24494206?v=4?s=100" width="100px;" alt="Akmal Soliev"/><br /><sub><b>Akmal Soliev</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=akmalsoliev" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nickto"><img src="https://avatars.githubusercontent.com/u/11967792?v=4?s=100" width="100px;" alt="Nick To"/><br /><sub><b>Nick To</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=nickto" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/kvnkho/"><img src="https://avatars.githubusercontent.com/u/32503212?v=4?s=100" width="100px;" alt="Kevin Kho"/><br /><sub><b>Kevin Kho</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=kvnkho" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yibenhuang"><img src="https://avatars.githubusercontent.com/u/62163340?v=4?s=100" width="100px;" alt="Yiben Huang"/><br /><sub><b>Yiben Huang</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=yibenhuang" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/andrewgross"><img src="https://avatars.githubusercontent.com/u/370118?v=4?s=100" width="100px;" alt="Andrew Gross"/><br /><sub><b>Andrew Gross</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=andrewgross" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/taniishkaaa"><img src="https://avatars.githubusercontent.com/u/109246904?v=4?s=100" width="100px;" alt="taniishkaaa"/><br /><sub><b>taniishkaaa</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=taniishkaaa" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://manuel.calzolari.name"><img src="https://avatars.githubusercontent.com/u/2764902?v=4?s=100" width="100px;" alt="Manuel Calzolari"/><br /><sub><b>Manuel Calzolari</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=manuel-calzolari" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
