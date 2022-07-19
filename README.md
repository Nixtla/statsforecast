# Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-13-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/indx_imgs/branding/logo_mid.png">
<h1 align="center">Statistical ⚡️ Forecast</h1>
<h3 align="center">Lightning fast forecasting with statistical and econometric models</h3>
    
[![CI](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Nixtla/statsforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/statsforecast)](https://pypi.org/project/statsforecast/)
[![PyPi](https://img.shields.io/pypi/v/statsforecast?color=blue)](https://pypi.org/project/statsforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/conda-forge/statsforecast?color=seagreen&label=conda)](https://anaconda.org/conda-forge/statsforecast)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Nixtla/statsforecast/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nixtla.github.io/statsforecast.svg?label=docs)](https://nixtla.github.io/statsforecast/)  
    
**StatsForecast** offers a collection of widely used univariate time series forecasting models, including automatic `ARIMA` and `ETS` modeling optimized for high performance using `numba`. It also includes a large battery of benchmarking models.
</div>

## 🔥  Highlights

* Fastest and most accurate `auto_arima` in `Python` and `R`.
* Fastest and most accurate `ets` in `Python` and `R`.

* **New!**: Replace FB-Prophet in two lines of code and gain speed and accuracy. Check the experiments [here](https://github.com/Nixtla/statsforecast/tree/main/experiments/arima_prophet_adapter).
* **New!**: Distributed computation in clusters with [ray](https://github.com/ray-project/ray). (Forecast 1M series in [30min](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray))
* **New!**: Good Ol' sklearn syntax with `AutoARIMA().fit(y).predict(h=7)`.

## 🎊 Features 

* Inclusion of `exogenous variables` and `prediction intervals` for ARIMA.
* 20x faster than `pmdarima`.
* 1.5x faster than `R`.
* 500x faster than `Prophet`. 
* Compiled to high performance machine code through [`numba`](https://numba.pydata.org/).
* 1,000,000 series in [30 min](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray) with [ray](https://github.com/ray-project/ray).

* Out of the box implementation of `ses`, `adida`, `historic_average`, `croston_classic`, `croston_sba`, `croston_optimized`, `seasonal_window_average`, `seasonal_naive`, `imapa`
`naive`, `random_walk_with_drift`, `window_average`, `seasonal_exponential_smoothing`, `tsb`, `auto_arima` and `ets`. 

Missing something? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## 📖 Why? 

Current Python alternatives for statistical models are slow, inaccurate and don't scale well. So we created a library that can be used to forecast in production environments or as benchmarks.  `StatsForecast` includes an extensive battery of models that can efficiently fit millions of time series.

## 🔬 Accuracy & ⏲ Speed 

### ARIMA 
The `auto_arima` model implemented in `StatsForecast` is **20x faster** than `pmdarima` and **1.5x faster** than `R`  while improving accuracy. You can see the exact comparison and reproduce the results [here](./experiments/arima/).

### ETS

The `ets` model implemented in `StatsForecast` is **4x faster** than `statsmodels` and *1.6x faster* than `R` while improving accuracy. You can see the exact comparison and reproduce the results [here](./experiments/ets/)

### Benchmarks at Scale

With `StatsForecast` you can fit 9 benchmark models on **1,000,000** series in under **5 min**. Reproduce the results [here](./experiments/benchmarks_at_scale/). 

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
conda install -c conda-forge statsforecast
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

## 🧬 Getting Started 
You can run this notebooks to get you started. 

* Example of different `auto_arima` models on M4 data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/statsforecast/blob/main/examples/arima.ipynb)  
    * In this notebook we present Nixtla's `auto_arima`. 
    The `auto_arima` model is widely used to forecast time series in production and as a benchmark. However, the alternative python implementation (`pmdarima`) is so slow that prevents data scientists from quickly iterating and deploying `auto_arima` in production for a large number of time series. 

* Shorter Example of fitting and `auto_arima` and an `ets` model.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/statsforecast/blob/main/examples/Forecasting_with_Auto_Arima_&_ETS_.ipynb) 


* Benchmarking 9 models on millions of [series](./experiments/benchmarks_at_scale/).

## 📖 Documentation (WIP)
Here is a link to the [documentation](https://nixtla.github.io/statsforecast/).

## 🔨 How to contribute
See [CONTRIBUTING.md](https://github.com/Nixtla/statsforecast/blob/main/CONTRIBUTING.md).

## 📃 References

*  The `auto_arima` model is based (translated) from the R implementation included in the [forecast](https://github.com/robjhyndman/forecast) package developed by Rob Hyndman.
*  The `ets` model is based (translated) from the R implementation included in the [forecast](https://github.com/robjhyndman/forecast) package developed by Rob Hyndman.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/FedericoGarza"><img src="https://avatars.githubusercontent.com/u/10517170?v=4?s=100" width="100px;" alt=""/><br /><sub><b>fede</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=FedericoGarza" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jmoralez"><img src="https://avatars.githubusercontent.com/u/8473587?v=4?s=100" width="100px;" alt=""/><br /><sub><b>José Morales</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=jmoralez" title="Code">💻</a> <a href="#maintenance-jmoralez" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/sugatoray/"><img src="https://avatars.githubusercontent.com/u/10201242?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sugato Ray</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=sugatoray" title="Code">💻</a></td>
    <td align="center"><a href="http://www.jefftackes.com"><img src="https://avatars.githubusercontent.com/u/9125316?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jeff Tackes</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Atackes" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/darinkist"><img src="https://avatars.githubusercontent.com/u/62692170?v=4?s=100" width="100px;" alt=""/><br /><sub><b>darinkist</b></sub></a><br /><a href="#ideas-darinkist" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/alech97"><img src="https://avatars.githubusercontent.com/u/22159405?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alec Helyar</b></sub></a><br /><a href="#question-alech97" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://dhirschfeld.github.io"><img src="https://avatars.githubusercontent.com/u/881019?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dave Hirschfeld</b></sub></a><br /><a href="#question-dhirschfeld" title="Answering Questions">💬</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mergenthaler"><img src="https://avatars.githubusercontent.com/u/4086186?v=4?s=100" width="100px;" alt=""/><br /><sub><b>mergenthaler</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=mergenthaler" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/kdgutier"><img src="https://avatars.githubusercontent.com/u/19935241?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kin</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=kdgutier" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Yasslight90"><img src="https://avatars.githubusercontent.com/u/58293883?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yasslight90</b></sub></a><br /><a href="#ideas-Yasslight90" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/asinig"><img src="https://avatars.githubusercontent.com/u/99350687?v=4?s=100" width="100px;" alt=""/><br /><sub><b>asinig</b></sub></a><br /><a href="#ideas-asinig" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/guerda"><img src="https://avatars.githubusercontent.com/u/230782?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Philip Gillißen</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=guerda" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/shagn"><img src="https://avatars.githubusercontent.com/u/16029092?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sebastian Hagn</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Ashagn" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
