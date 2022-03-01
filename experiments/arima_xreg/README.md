# ARIMA with exogenous regressors 

## Main results

| metric (mean)   |   auto_arima_nixtla |   auto_arima_pmdarima |   auto_arima_r |
|:---------|--------------------:|----------------------:|---------------:|
| mase     |                0.84 |                  0.89 |           0.82 |
| time (mins)|                3.32 |                 52.51 |           2.91 |

## Reproducibility

To reproduce the main results you have:

1. Execute `make init` to create a Docker image with the required dependencies.
2. Run the experiments using `make run_module module="python -m src.[model] --group [group]"` where `[model]` can be `statsforecast` and `pmdarima`, and `[group]` can be `NP`, `PJM`, `FR`, `BE` y `DE`.
3. To run R experiments you have to prepare the data using `make run_module module="python -m src.data --group [group]"` for each `[group]`. Once it is done, just run `make run_module module="Rscript src/arima_r.R [group]"`.
4. Finally you can evaluate the forecasts using `make run_module module="python -m src.evaluation"`.
