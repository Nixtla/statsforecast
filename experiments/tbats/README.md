# TBATS: A brand-new implementation for the Python ecosystem

## TL;DR

StatsForecast's new implementation of TBATS outperforms current implementations in terms of speed with comparable accuracy. 

## Intro 

TBATS, developed by [De Livera, Hyndman, and Snyder (2011)](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf), is an innovations state space model used for forecasting time series with multiple seasonalities. The model employs a combination of Fourier terms to represent the seasonal patterns, a Box-Cox transformation, and ARMA errors. The acronym TBATS stands for **T**rigonometric, **B**ox-Cox transformation, **A**RMA errors, **T**rend, and **S**easonal components.

The first implementation of the TBATS model was in the [forecast](https://pkg.robjhyndman.com/forecast/reference/tbats.html) R package, and until now, the only Python implementation of this model was [tbats](https://github.com/intive-DataScience/tbats). A well-known drawback of both implementations is that they can be very slow for long time series. StatsForecast has addressed this issue by introducing an improved version of TBATS that enhances speed while maintaining similar accuracy. This new version automatically selects the optimal number of Fourier terms using the method proposed by [De Livera et al.](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf) in their 2011 paper. This method, not implemented in either the R or the Python versions, reduces the total execution time, marking a considerable improvement over the other two implementations.

Two versions of the model have been included in StatsForecast: `AutoTBATS` and `TBATS`. The former automatically tests all feasible combinations of the parameters `use_box_cox`, `use_trend`, `use_damped_trend`, and `use_arma_errors`, selecting the model with the lowest AIC, while the latter only generates the model specified by the user's parameters.

TBATS excels in analyzing time series with multiple seasonalities. In fact, it was developed to overcome the limitations of traditional models like ETS or ARIMA, which can only handle one seasonal pattern, and thus, have a more limited modeling capability. 

## Experiment 

StatsForecast's `AutoTBATS` was evaluated using the [M3](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000571) and the [M4](https://www.sciencedirect.com/science/article/pii/S0169207019301128) Competition datasets, comparing its performance in accuracy and time with the existing implementations. 

The M3 dataset contains 3003 time series, with the following frequencies and seasonal periods. 

![m3_description](https://github.com/Nixtla/statsforecast/assets/47995617/82760595-e240-47f1-b56b-b8d5c2ee0634)

The M4 dataset contains 100,000 time series, with the following frequencies and seasonal periods.

![m4_description](https://github.com/Nixtla/statsforecast/assets/47995617/374287a2-ab62-47ae-b91a-b5dd1f8db8b1)

We refer to the R version as `R-TBATS` and the Python version as `PY-TBATS`. We opted for `AutoTBATS` instead of `TBATS` because the former is what the current R and Python versions do: unless otherwise specified by the user, automatically test all feasible combinations of the parameters and then select the model with the lowest AIC. Additionally, our comparison included a [Seasonal Naive model](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#seasonalnaive), which uses the value of the last seasonal period, as a baseline. 

Notice that in the M4 dataset, the hourly frequency has multiple seasonalities, namely 24 (a day) and 24*7 (a week). For `AutoTBATS`, `R-TBATS` and `PY-TBATS`, we used both seasonalities when generating the forecasts. 

### Perfomance evaluation

To evaluate the accuracy of the forecasts, we used common performance metrics: the [Mean Average Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#mae) (MAE), the [Root Mean Squared Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#rmse) (RMSE), the [Mean Absolute Percentage Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#mape) (MAPE), and the [symmetric Mean Absolute Percentage Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#smape) (sMAPE).

The forecast horizons were the same as in the M3 and M4 competitions, shown in the previous tables. 

### Results

For the M3 dataset, we obtained the following results: 

![m3_accuracy](https://github.com/Nixtla/statsforecast/assets/47995617/e2a07768-7fc8-409f-81e8-ef28bd8856b7)

![m3_time](https://github.com/Nixtla/statsforecast/assets/47995617/c84000b2-f3e8-4225-940c-6c27b8fc1c50)

Given the long execution time of the `PY-TBATS` implemenatation, we decided to exclude it from the experiment with the M4 dataset. 

For the M4 dataset, we obtained the following results:  

![m4_accuracy](https://github.com/Nixtla/statsforecast/assets/47995617/483144a8-4eb9-42f4-91a5-cfaf08022b45)

![m4_time](https://github.com/Nixtla/statsforecast/assets/47995617/2e010aab-ee42-47fd-b85e-b46091f0f7ca)

### Conclusion

In this experiment, StatsForecast's `AutoTBATS` demonstrated similar performance to the R and Python implementations in terms of accuracy across several metrics. In terms of speed, it demonstrated equal or superior performance compared to the other two.

  - For the complete M3 dataset, StatsForecast's `AutoTBATS` is almost 30x faster than the current Python implementation, with very similar accuracy. 
  - For the complete M3 dataset, StatsForecast's `AutoTBATS` is 1.6x faster than the R implementation. For the largest frequency in this dataset (Monthly), StatsForecast's `AutoTBATS` is 2.3x faster than R. 
  - Similarly, for the complete M4 dataset, StatsForecast's `AutoTBATS` is 2.5x faster than the R implementation. For the largest frequency in this dataset (Monthly), StatsForecast's `AutoTBATS` is 2.7x faster than R. 
  - In the cases where StatsForecast's `AutoTBATS` is slower than R, the difference is relatively small.

As a result, StatsForecast's `AutoTBATS` represents a competitive Python implementation of the TBATS model, maintaining similar accuracy to both the R and Python versions. Notably, it is as fast, if not faster than, the R implementation and considerably outpaces the current Python model in terms of speed. Therefore, `AutoTBATS` should be regarded as a viable option within a forecasting pipeline, either to be used alongside other models from StatsForecast or as a baseline.

### Reproducibility
1. Create a conda environment `exp_tbats` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

2. Activate the conda environment using
  ```shell
  conda activate exp_tbats
  ```

3. Run the experiments. When running the Python scripts for StatsForecast, you need to specify the dataset, group, and model (`AutoTBATS` or `SeasonalNaive`). 
  ```shell
  python -m data --dataset=dataset --group=group # generates train set
  python -m data --dataset=dataset --group=group --train=False # generates test set 
  python -m experiment --dataset=dataset --group=group --model=model 
  python -m py_tbats --dataset=dataset --group=group
  Rscript r_tbats.R # select dataset and group inside script
  ```
  
4. Once the experiments for each dataset are complete, evaluate the forecasts. 
  ```shell
  python -m evaluation --dataset=dataset
  ```

### References

[De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. J American Statistical Association, 106(496), 1513–1527.](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf)
