# TBATS: A brand-new implementation for the Python ecosystem

TBATS, developed by De Livera, Hyndman, and Snyder (2011), is an innovations state space model used for forecasting time series with multiple seasonalities. It uses a combination of Fourier terms to represent the seasonal patterns, a Box-Cox transformation, and ARMA errors. 

The first implementation of TBATS was in the [forecast](https://pkg.robjhyndman.com/forecast/reference/tbats.html) R package. Until now, the only Python implementation of this model was [tbats](https://github.com/intive-DataScience/tbats). A well-known drawback of both implementations is that they can be very slow for long time series. 

StatsForecast's new TBATS improves the performance of current implementations, exceeding the accuracy of the R version and easily surpassing `tbats` in terms of total execution time. A standout feature of this new version is its automatic selection of the optimal number of Fourier terms. This method, proposed in the 2011 paper by De Livera et al., was not incorporated in previous implementations, but it considerably speeds up the model. 

Two versions of the model have been included in StatsForecast: `AutoTBATS` and `TBATS`. The former automatically tests all feasible combinations of the parameters `use_box_cox`, `use_trend`, `use_damped_trend`, and `use_arma_errors`, selecting the model with the lowest AIC, while the latter only generates the model specified by the user's parameters.

## Experiment 

TBATS was tested using a dataset that contains 32,896 observations of hourly electricity consumption. This type of data contains two seasonal periods: 24 (a day) and 24*7 (a week). We used cross-validation with five windows of 24 hours. We also included two baselines: StatsForecast's Seasonal Naive and Prophet. 

## Results 

** Total execution time **
| Model         | Time (min) |
|---------------|------------|
|AutoTBATS      | 9.43      |
|TBATS-R        | 36.08      |
|TBATS-PY       | 60.97      |
|Seasonal Naive | 0.03       |
|Prophet        | 0.80       |

** Accuracy **
| Metric | AutoTBATS | TBATS-R | TBATS-PY | Seasonal Naive | Prophet |
|--------|-----------|---------|----------|----------------|---------|
| MAE    | 1535.31   | 1932.32 | 3483.42  | 1593.33        | 2338.43 |
| RMSE   | 1771.48   | 2204.27 | 4583.29  | 1863.48        | 2735.65 |


** Notes ** 
- For TBATS-PY cross-validation was done using only one window of 24 hours due to the long execution time. 
- For both MAE and RMSE, the accuracy reported is the mean accuracy across the 5 windows, except in the case of TBATS-PY, where only one window was used. 

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

De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. J American Statistical Association, 106(496), 1513–1527. 
