# TBATS: A brand-new implementation for the Python ecosystem

## TL;DR 
For next-day forecasting of electricity load, StatsForecast's new implementation of TBATS outperforms current implementations of this model in both accuracy and speed.

## Intro 
TBATS, developed by [De Livera, Hyndman, and Snyder (2011)](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf), is an innovations state space model used for forecasting time series with multiple seasonalities. It uses a combination of Fourier terms to represent the seasonal patterns, a Box-Cox transformation, and ARMA errors. 

The first implementation of TBATS was in the [forecast](https://pkg.robjhyndman.com/forecast/reference/tbats.html) R package, and until now, the only Python implementation of this model was [tbats](https://github.com/intive-DataScience/tbats). A well-known drawback of both implementations is that they can be very slow for long time series. 

StatsForecast's new TBATS improves the performance of these two implementations, exceeding both in accuracy and speed. A standout feature of this new version is its automatic selection of the optimal number of Fourier terms. The method to do this was proposed in the 2011 paper by [De Livera et al.](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf), but it was not incorporated in any of the current implementations although it considerably reduces the total execution time. 

Two versions of the model have been included in StatsForecast: `AutoTBATS` and `TBATS`. The former automatically tests all feasible combinations of the parameters `use_box_cox`, `use_trend`, `use_damped_trend`, and `use_arma_errors`, selecting the model with the lowest AIC, while the latter only generates the model specified by the user's parameters.

TBATS excels in analyzing time series with multiple seasonalities, such as hourly electricity data showing daily and weekly patterns. In fact, it was developed to overcome the limitations of traditional models like ETS or ARIMA, which can only handle one seasonal pattern, and thus, have a more limited modeling capacity.

## Experiment 

StatsForecast's `AutoTBATS` was evaluated using a dataset of 32,896 hourly electricity consumption observations, featuring daily (24-hour) and weekly (168-hour) seasonal periods. This data, sourced from PJM Interconnection LLC, a US regional transmission organization, is available [here](https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv.). 

In our analysis, we benchmarked StatsForecast's implementation against the two existing versions. We refer to the R version as `TBATS-R` and the Python version as `TBATS-PY`. We opted for `AutoTBATS` instead of `TBATS` because the former is what the current R and Python versions do: unless otherwise specified by the user, automatically test all feasible combinations of the parameters and then select the model with the lowest AIC. Additionally, our comparison included a [Seasonal Naive model](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#seasonalnaive), which uses the value of the last seasonal period, and [Prophet](https://facebook.github.io/prophet/), an additive model designed to handle complex seasonality and holiday effects. 


### Perfomance evaluation 

We used cross-validation to evaluate the accuracy of the models, focusing on two key error metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). In time series data, cross-validation is implemented by defining a sliding window across the historical data, followed by predicting the subsequent period. Here we used 5 validation windows, each spanning 24 hours, to ensure a thorough and accurate assessment. The error metrics are averaged across each window to derive the final score. For a more detailed understanding of cross-validation in this context, refer to the StatsForecast's [documentation](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html).

## Results 

Using one cross-validation window, we obtained the following results. 

![tbats_results1](https://github.com/Nixtla/statsforecast/assets/47995617/41e2f738-011e-4299-9c9b-8a1913ab6f06)

Given the long execution time of the Python version, we decided to exclude it from the experiment with 5 cross-validation windows. Here we also included the Seasonal Naive model and Prophet as baselines. 

![tbats_results5](https://github.com/Nixtla/statsforecast/assets/47995617/e73a7ca6-132e-45c8-bdfa-e4f0aa412eec)

## Conclusion 

In this experiment, StastsForecast's new TBATS implementation demonstrated superior performance over the existing R and Python versions in both accuracy and speed.

- The new model is significantly faster, being almost 4 times faster than the R version and 30 times faster than the Python version.
- In terms of accuracy, it outperformed the R version by 10 to 20%, which itself already surpasses the Python version.

Moreover, this implementation has proven more accurate than both the Seasonal Naive model and Prophet, proving its effectiveness as a forecasting method for time series data with multiple seasonalities.

Looking ahead, we plan to conduct a more comprehensive analysis of this model's performance across various datasets, which will provide deeper insights into its capabilities and potential applications.

## Reproducibility

1. Create a conda environment `exp_tbats` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate exp_tbats
  ```

4. Run the experiments for each dataset and each model using 
  ```shell
  Rscript src/tbats_r.R
  python -m src.tbats_py
  python -m src.main
  ```

## References 

[De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. J American Statistical Association, 106(496), 1513–1527.](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf)


## M4 Hourly Results
| Model    | SMAPE | Time (minutes) |
| -------- | ----- | -------------- |
| AutoTBATS| 12.4% | 31.8           |
| TBATS (R)| 12.8% | 128.3          |

To replicate:
```shell
python m4/tbats.py
Rscript m4/tbats.R
```
