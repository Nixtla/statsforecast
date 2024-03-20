# TBATS: A brand-new implementation for the Python ecosystem

## TL;DR

StatsForecast's new implementation of TBATS outperforms the R implementation in terms of speed with comparable accuracy. 

## Intro 

TBATS, developed by [De Livera, Hyndman, and Snyder (2011)](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf), stands for **T**rigonometric, **B**ox-Cox transformation, **A**RMA errors, **T**rend, and **Seasonal components. It is an innovations state space model used for forecasting time series with multiple seasonalities. The model employs a combination of Fourier terms to represent the seasonal patterns, a Box-Cox transformation, and ARMA errors.

TBATS, developed by [De Livera, Hyndman, and Snyder (2011)](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf), is an innovations state space model used for forecasting time series with multiple seasonalities. It uses a combination of Fourier terms to represent the seasonal patterns, a Box-Cox transformation, and ARMA errors. 

The original implementation of the TBATS model was in the [forecast](https://pkg.robjhyndman.com/forecast/reference/tbats.html) R package, and is well-known for its slow performance with long time series. StatsForecast has addressed this issue by introducing an improved version of TBATS that enhances speed while maintaining similar accuracy. A notable advancement in this new version is the automatic selection of the optimal number of Fourier terms, a significant improvement proposed by [De Livera et al.](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf) in their 2011 paper. This method, not implemented in the R version, reduces the total execution time, marking a considerable improvement over the original model.

Two versions of the model have been included in StatsForecast: `AutoTBATS` and `TBATS`. The former automatically tests all feasible combinations of the parameters `use_box_cox`, `use_trend`, `use_damped_trend`, and `use_arma_errors`, selecting the model with the lowest AIC, while the latter only generates the model specified by the user's parameters.

TBATS excels in analyzing time series with multiple seasonalities. In fact, it was developed to overcome the limitations of traditional models like ETS or ARIMA, which can only handle one seasonal pattern, and thus, have a more limited modeling capacity.

## Experiment 

StatsForecast's `AutoTBATS` was evaluated using the [M3](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000571) and the [M4](https://www.sciencedirect.com/science/article/pii/S0169207019301128) Competition datasets, comparing its performance in accuracy and time with the existing R implementation. 

The M3 dataset contains 3003 time series, with the following frequencies and seasonal periods. 

![m3_description](https://github.com/Nixtla/statsforecast/assets/47995617/95cbc7a4-593c-4252-b943-e97040f7a32b)

The M4 dataset contains 100,000 time series, with the following frequencies and seasonal periods.

![m4_description](https://github.com/Nixtla/statsforecast/assets/47995617/70a4b280-193f-4881-bbd6-21063ac5a86b)

Notice that in this dataset, the hourly frequency has multiple seasonalities, namely 24 (a day) and 168 (a week). Both StatsForecast and R used both seasonalities when generating the forecasts. We also included the `SeasonalNaive` model from StatsForecast as a baseline. 

### Perfomance evaluation

To evaluate the accuracy of the forecasts, we used common performance metrics: the [Mean Average Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#mae) (MAE), the [Root Mean Squared Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#rmse) (RMSE), the [Mean Absolute Percentage Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#mape) (MAPE), and the [symmetric Mean Absolute Percentage Error](https://nixtlaverse.nixtla.io/utilsforecast/losses.html#smape) (sMAPE).

The forecast horizons were the same as in the M3 and M4 competitions, shown in the previous tables. 

### Results

For the M3 dataset, we obtained the following results: 

![m3_accuracy](https://github.com/Nixtla/statsforecast/assets/47995617/79936ad8-fd07-4c29-aaf1-e900946c1b79)

![m3_time](https://github.com/Nixtla/statsforecast/assets/47995617/3ef6e360-ad08-46af-aa22-666a7f63c5f6)

For the M4 dataset, we obtained the following results:  

![m4_accuracy](https://github.com/Nixtla/statsforecast/assets/47995617/94c6099a-e9c4-4c7f-9343-8729abfcf534)

![m4_time](https://github.com/Nixtla/statsforecast/assets/47995617/c9da434b-2334-476c-9531-ed8395cdaf30)

### Conclusion

In this experiment, StatsForecast's `AutoTBATS` implementation demonstrated similar performance to the R implementation in terms of accuracy across several metrics. In terms of speed, it demonstrated equal or superior performance compared to R's.

  - For the largest group in the M3 dataset (Monthly), StatsForecast's `AutoTBATS` is 3x faster than the R implementation. For the second largest group (Quarterly), it is 1.5x faster than R.
  - Similarly, for the largest group in the M4 dataset (Monthly), StatsForecast's `AutoTBATS` is 3.2x faster than the R implementation. For the second largest group (Quarterly), it is almost 2x faster than R.
  - In the cases where StatsForecast's `AutoTBATS` is slower than R, the difference is relatively small.

As a result, we can conclude that StatsForecast's `AutoTBATS` is a competitive Python implementation of the TBATS model, with similar accuracy and superior speed compared to the R implementation. Hence, it should be considered as a viable option that can be used as part of a forecasting pipeline, alongside other models from StatsForecast, or as a baseline.

### Reproducibility
1. Create a conda environment `exp_tbats` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using
  ```shell
  conda activate exp_tbats
  ```

4. Run the experiments for each dataset and for each group
  ```shell
  python -m data --dataset=dataset --group=group 
  python -m data --dataset=dataset --group=group --train=False
  python -m experiment --dataset=dataset --group=group
  Rscript r_tbats.R
  ```
  
5. Evaluate the forecasts 
  ```shell
  python -m evaluation --dataset=dataset
  ```

### References

[De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. J American Statistical Association, 106(496), 1513–1527.](https://www.robjhyndman.com/papers/ComplexSeasonality.pdf)
