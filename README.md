# Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-21-orange.svg?style=flat-square)](#contributors-)
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
    
**StatsForecast** offers a collection of widely used univariate time series forecasting models, including automatic `ARIMA` and `ETS` modeling optimized for high performance using `numba`. It also includes a large battery of benchmarking models.
</div>

## 💻 Installation
<details open>
<summary>PyPI</summary>

You can install the *released version* of `StatsForecast` from the Python package index [pip](https://pypi.org/project/statsforecast/) with:

```python
pip install statsforecast
```

(Installing inside a python virtualenvironment or a conda environment is recommended.)
</details>

<details open>
<summary>Conda</summary>
  
Also you can install the *released version* of `StatsForecast` from [conda](https://anaconda.org/conda-forge/statsforecast) with:

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

## 🏃🏻‍♀️🏃 Getting Started
To get started just follow this [guide](https://colab.research.google.com/drive/1GKoLXb5KENLPMuSE9torLGvKPCyXh-Cl?usp=sharing).
In the guide, we showcase `AutoARIMA` and `AutoETS`, and go further into probabilistic predictions, exogenous variables, and other [baseline models](https://nixtla.github.io/statsforecast/models.html).

## 🎉 New!
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/statsforecast/blob/main/nbs/examples/Getting_Started_with_Auto_Arima_and_ETS.ipynb) **ETS Example**: 4x faster than StatsModels with improved accuracy and robustness.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/statsforecast/blob/main/nbs/examples/AutoArima_vs_Prophet.ipynb)   **Complete pipeline and comparison**:  20x faster than pmdarima and 500x faster than Prophet. 

## 🔥  Highlights

* Fastest and most accurate `AutoARIMA` in `Python` and `R`.
* Fastest and most accurate `ETS` in `Python` and `R`.

* Replace FB-Prophet in two lines of code and gain speed and accuracy. Check the experiments [here](https://github.com/Nixtla/statsforecast/tree/main/experiments/arima_prophet_adapter).
* Distributed computation in clusters with [ray](https://github.com/ray-project/ray). (Forecast 1M series in [30min](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray))
* Good Ol' sklearn interface with `AutoARIMA().fit(y).predict(h=7)`.

## 🎊 Features 

* Inclusion of `exogenous variables` and `prediction intervals` for ARIMA.
* 20x faster than `pmdarima`.
* 1.5x faster than `R`.
* 500x faster than `Prophet`.
* 100x faster than `NeuralProphet`.
* 4x faster than `statsmodels`.
* Compiled to high performance machine code through [`numba`](https://numba.pydata.org/).
* 1,000,000 series in [30 min](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray) with [ray](https://github.com/ray-project/ray).

* Out of the box implementation of `ADIDA`, `HistoricAverage`, `CrostonClassic`, `CrostonSBA`, `CrostonOptimized`, `SeasonalWindowAverage`, `SeasonalNaive`, `IMAPA`
`Naive`, `RandomWalkWithDrift`, `WindowAverage`, `SeasonalExponentialSmoothing`, `TSB`, `AutoARIMA` and `ETS`.

Missing something? Please open an issue or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## 📖 Why? 

Current Python alternatives for statistical models are slow, inaccurate and don't scale well. So we created a library that can be used to forecast in production environments or as benchmarks.  `StatsForecast` includes an extensive battery of models that can efficiently fit millions of time series.

## 🔬 Accuracy & ⏲ Speed 

### ARIMA 
The `AutoARIMA` model implemented in `StatsForecast` is **20x faster** than `pmdarima` and **1.5x faster** than `R`  while improving accuracy. You can see the exact comparison and reproduce the results [here](./experiments/arima/).

### ETS

StatsForecast's exponential smoothing is **4x faster than StatsModels'** and **1.6x faster than R's**, with improved accuracy and robustness. You can see the exact comparison and reproduce the results [here](./experiments/ets/)

### Benchmarks at Scale

With `StatsForecast` you can fit 9 benchmark models on **1,000,000** series in under **5 min**. Reproduce the results [here](./experiments/benchmarks_at_scale/). 



## 🧬 Getting Started 
You can run this notebooks to get you started. 

* Example of different `AutoARIMA` models on M4 data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/statsforecast/blob/main/nbs/examples/AutoArima_vs_Prophet.ipynb)   
    * In this notebook we present Nixtla's `AutoARIMA`. 
    The `AutoARIMA` model is widely used to forecast time series in production and as a benchmark. However, the alternative python implementation (`pmdarima`) is so slow that prevents data scientists from quickly iterating and deploying `AutoARIMA` in production for a large number of time series.

* Shorter Example of fitting and `AutoARIMA` and an `ETS` model.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/statsforecast/blob/main/nbs/examples/Getting_Started_with_Auto_Arima_and_ETS.ipynb) 


* Benchmarking 9 models on millions of [series](./experiments/benchmarks_at_scale/).

## 📖 Documentation (WIP)
Here is a link to the [documentation](https://nixtla.github.io/statsforecast/).

## 🔨 How to contribute
See [CONTRIBUTING.md](https://github.com/Nixtla/statsforecast/blob/main/CONTRIBUTING.md).

## 📃 References

*  The `AutoARIMA` model is based (translated) from the R implementation included in the [forecast](https://github.com/robjhyndman/forecast) package developed by Rob Hyndman.
*  The `ETS` model is based (translated) from the R implementation included in the [forecast](https://github.com/robjhyndman/forecast) package developed by Rob Hyndman.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/FedericoGarza"><img src="https://avatars.githubusercontent.com/u/10517170?v=4?s=100" width="100px;" alt="fede"/><br /><sub><b>fede</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=FedericoGarza" title="Code">💻</a> <a href="#maintenance-FedericoGarza" title="Maintenance">🚧</a></td>
      <td align="center"><a href="https://github.com/jmoralez"><img src="https://avatars.githubusercontent.com/u/8473587?v=4?s=100" width="100px;" alt="José Morales"/><br /><sub><b>José Morales</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=jmoralez" title="Code">💻</a> <a href="#maintenance-jmoralez" title="Maintenance">🚧</a></td>
      <td align="center"><a href="https://www.linkedin.com/in/sugatoray/"><img src="https://avatars.githubusercontent.com/u/10201242?v=4?s=100" width="100px;" alt="Sugato Ray"/><br /><sub><b>Sugato Ray</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=sugatoray" title="Code">💻</a></td>
      <td align="center"><a href="http://www.jefftackes.com"><img src="https://avatars.githubusercontent.com/u/9125316?v=4?s=100" width="100px;" alt="Jeff Tackes"/><br /><sub><b>Jeff Tackes</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Atackes" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/darinkist"><img src="https://avatars.githubusercontent.com/u/62692170?v=4?s=100" width="100px;" alt="darinkist"/><br /><sub><b>darinkist</b></sub></a><br /><a href="#ideas-darinkist" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="https://github.com/alech97"><img src="https://avatars.githubusercontent.com/u/22159405?v=4?s=100" width="100px;" alt="Alec Helyar"/><br /><sub><b>Alec Helyar</b></sub></a><br /><a href="#question-alech97" title="Answering Questions">💬</a></td>
      <td align="center"><a href="https://dhirschfeld.github.io"><img src="https://avatars.githubusercontent.com/u/881019?v=4?s=100" width="100px;" alt="Dave Hirschfeld"/><br /><sub><b>Dave Hirschfeld</b></sub></a><br /><a href="#question-dhirschfeld" title="Answering Questions">💬</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/mergenthaler"><img src="https://avatars.githubusercontent.com/u/4086186?v=4?s=100" width="100px;" alt="mergenthaler"/><br /><sub><b>mergenthaler</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=mergenthaler" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/kdgutier"><img src="https://avatars.githubusercontent.com/u/19935241?v=4?s=100" width="100px;" alt="Kin"/><br /><sub><b>Kin</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=kdgutier" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/Yasslight90"><img src="https://avatars.githubusercontent.com/u/58293883?v=4?s=100" width="100px;" alt="Yasslight90"/><br /><sub><b>Yasslight90</b></sub></a><br /><a href="#ideas-Yasslight90" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="https://github.com/asinig"><img src="https://avatars.githubusercontent.com/u/99350687?v=4?s=100" width="100px;" alt="asinig"/><br /><sub><b>asinig</b></sub></a><br /><a href="#ideas-asinig" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center"><a href="https://github.com/guerda"><img src="https://avatars.githubusercontent.com/u/230782?v=4?s=100" width="100px;" alt="Philip Gillißen"/><br /><sub><b>Philip Gillißen</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=guerda" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/shagn"><img src="https://avatars.githubusercontent.com/u/16029092?v=4?s=100" width="100px;" alt="Sebastian Hagn"/><br /><sub><b>Sebastian Hagn</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Ashagn" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/fugue-project/fugue"><img src="https://avatars.githubusercontent.com/u/21092479?v=4?s=100" width="100px;" alt="Han Wang"/><br /><sub><b>Han Wang</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=goodwanghan" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://www.linkedin.com/in/benjamin-jeffrey-218548a8/"><img src="https://avatars.githubusercontent.com/u/36240394?v=4?s=100" width="100px;" alt="Ben Jeffrey"/><br /><sub><b>Ben Jeffrey</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Abjeffrey92" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/Beliavsky"><img src="https://avatars.githubusercontent.com/u/38887928?v=4?s=100" width="100px;" alt="Beliavsky"/><br /><sub><b>Beliavsky</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=Beliavsky" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/MMenchero"><img src="https://avatars.githubusercontent.com/u/47995617?v=4?s=100" width="100px;" alt="Mariana Menchero García "/><br /><sub><b>Mariana Menchero García </b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=MMenchero" title="Code">💻</a></td>
      <td align="center"><a href="https://www.linkedin.com/in/guptanick/"><img src="https://avatars.githubusercontent.com/u/33585645?v=4?s=100" width="100px;" alt="Nikhil Gupta"/><br /><sub><b>Nikhil Gupta</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Angupta23" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/jdegene"><img src="https://avatars.githubusercontent.com/u/17744939?v=4?s=100" width="100px;" alt="JD"/><br /><sub><b>JD</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/issues?q=author%3Ajdegene" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://github.com/jattenberg"><img src="https://avatars.githubusercontent.com/u/924185?v=4?s=100" width="100px;" alt="josh attenberg"/><br /><sub><b>josh attenberg</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=jattenberg" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/JeroenPeterBos"><img src="https://avatars.githubusercontent.com/u/15342738?v=4?s=100" width="100px;" alt="JeroenPeterBos"/><br /><sub><b>JeroenPeterBos</b></sub></a><br /><a href="https://github.com/Nixtla/statsforecast/commits?author=JeroenPeterBos" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
