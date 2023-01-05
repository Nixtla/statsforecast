# Probabilistic forecasting with StatsForecast's AutoETS is accurate and faster than other implementations  

## Introduction 

[AutoETS](https://nixtla.github.io/statsforecast/models.html#autoets) is a fast and reliable model in [StatsForecast](https://nixtla.github.io/statsforecast/) that automatically selects the best [ETS model](https://otexts.com/fpp3/ets.html) for a given time series. It is a probabilistic model and it can generate prediction intervals for any confidence level. In this experiment, we'll show that these prediction intervals are accurate and fast when compared to two well-known forecasting libraries: R's [forecast](https://pkg.robjhyndman.com/forecast/) package and [statsmodels](https://www.statsmodels.org/dev/index.html). 

## Experiment 

We'll use the [M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) dataset, which consists of 100,000 time series of different frequencies. Each frequency has a different forecast horizon. 

| dataset  |Number of series | Forecast horizon| 
|:---------|:---------------:|:---------------:|
| Hourly   | 414             | 48              |
| Daily    | 4227            | 14              |
| Weekly   | 359             | 13              |
| Monthly  | 48000           | 18              |
| Quarterly| 24000           | 8               |
| Yearly   | 23000           | 6               |

Our goal is to generate prediction intervals for all time series with the following confidence levels: 55, 60, 65, 70, 75, 80, 85, 90, and 95%.

As stated in the introduction, this experiment will test the performance in accuracy and time of 
1. StatsForecast 
2. R's forecast package 
3. statsmodels

To measure the accuracy of the prediction intervals, we'll use the [Winkler score](https://otexts.com/fpp3/distaccuracy.html). The final score for each frequency is computed using the mean or the median of the Winkler scores of all time series. Conventionally, the mean is used, but here we also included the median to eliminate the extreme cases found in statsmodels. 

Both StatsForecast and R can automatically select the best ETS model for a given time series. Here by "best" we mean the model with the lowest Akaike Information Criterion (AICc). statsmodels doesn't have this functionality, so we first find the best model, and then we generate their prediction intervals. In all cases, if no ETS model can be generated, then a [naive](https://otexts.com/fpp3/simple-methods.html) model is used. 

## Results 

| dataset   | metric                 |     ets_r |   ets_statsforecast |   ets_statsmodels |
|:----------|:-----------------------|----------:|--------------------:|------------------:|
| Hourly    | Winkler-score (mean)   | 162580    |           163823    |       5.40095e+12 |
| Hourly    | Winkler-score (median) |   4304.54 |             4398.64 |    4411.2         |
| Hourly    | time                   |    219.4  |              159.67 |     761.04        |
| Daily     | Winkler-score (mean)   |  71494.9  |            71321.9  |  137536           |
| Daily     | Winkler-score (median) |  60436    |            60374.7  |   58964.2         |
| Daily     | time                   |    114.08 |              211.63 |     145.82        |
| Weekly    | Winkler-score (mean)   |  54841.8  |            54899.7  |   54301.9         |
| Weekly    | Winkler-score (median) |  32359.7  |            31305.2  |   31154.2         |
| Weekly    | time                   |      4.18 |               25.55 |       8.53        |
| Monthly   | Winkler-score (mean)   |  39853.4  |            40626    |       5.0763e+16  |
| Monthly   | Winkler-score (median) |  25767.7  |            26686.4  |   26508.3         |
| Monthly   | time                   |   8320.72 |             3708.56 |   18464.3         |
| Quarterly | Winkler-score (mean)   |  51232    |            51772.9  |       1.45623e+20 |
| Quarterly | Winkler-score (median) |  37028.8  |            37986.2  |   37275.2         |
| Quarterly | time                   |    706.95 |              601.32 |    2549.96        |
| Yearly    | Winkler-score (mean)   |  48929.8  |            49102.2  |       1.241e+07   |
| Yearly    | Winkler-score (median) |  32903.8  |            33339.2  |   34232.6         |
| Yearly    | time                   |    315.47 |               50.8  |     395.92        |

*Note*: Time is measured in seconds

## Reproducibility 

1. Execute `conda env create -f environment.yml`. 
2. Activate the environment using `conda activate ets_intervals`. 
3. Prepare the data for each frequency using `python -m src.data --dataset M4 --group [group]`. Here `[group]` can be `Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, or `Yearly`. 
4. Run the StatsForecast experiments with `python -m src.nb_statsforecast --datasets M4 --group [group]`.
5. Run the statsmodels experiments with `python -m src.nb_statsmodels --datasets M4 --group [group]`. 
6. Run the R experiments with `Rscript src/ets_r.R [group]`. 
7. Evaluate the prediction intervals using `python -m src.evaluation.py`

## Conclusions 

- StatsForecast's AutoETS produces prediction intervals that are close in accuracy to the ones produced by R's forecast library and by statsmodels. 
- For most frequencies, StatsForecast is faster than R and statsmodels. In the monthly frequency, which corresponds to 48% of the M4 Competition dataset, StatsForecast is 2.2x faster than R and 5x faster than statsmodels. 
- In terms of accuracy, StatsForecast's results are not surprising since AutoETS is a mirror of R's ets function. However, in terms of time, StatsForecast's performance is superior to both R and statsmodels. 
- StatsForecast has multiple probabilistic models in addition to AutoETS. See the full list [here.](https://nixtla.github.io/statsforecast/examples/models_intro.html). 

## References 

- StatsForecast [quick start guide.](https://nixtla.github.io/statsforecast/examples/getting_started_short.html) 
- A tutorial on probabilistic forecasting using StatsForecast can be found [here.](https://nixtla.github.io/statsforecast/examples/uncertaintyintervals.html)
