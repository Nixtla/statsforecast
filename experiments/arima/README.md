# ARIMA experiments

## Main results

| dataset   | metric   |   auto_arima_nixtla | auto_arima_pmdarima [1] |   auto_arima_r |   prophet |
|:----------|:---------|--------------------:|----------------------:|---------------:|----------:|
| Daily     | MASE     |                **3.26** |                  3.35 |           4.46 |     14.26 |
| Daily     | time     |                **1.41** |                 27.61 |           1.81 |    514.33 |
| Hourly    | MASE     |                **0.92** |                ---    |           1.02 |      1.78 |
| Hourly    | time     |               **12.92** |                ---    |          23.95 |     17.27 |
| Weekly    | MASE     |                **2.34** |                  2.47 |           2.58 |      7.29 |
| Weekly    | time     |                0.42 |                  2.92 |           **0.22** |     19.82 |


[1] The model `auto_arima` from `pmdarima` had a problem with Hourly data. An issue was opened.

## Reproducibility

To reproduce the main results you have:

1. Execute `make init` to create a Docker image with the needed dependencies.
2. Run the experiments using `make run_module module="python -m src.[model] --dataset M4 --group [group]"` where `[model]` can be `statsforecast`, `pmdarima` and `prophet`, and `[group]` can be `Daily`, `Hourly` and `Weekly`.
3. To run R experiments you have to prepare the data using `make run_module module="python -m src.data --dataset M4 --group [group]"` for each `[group]`. Once it is done, just run `make run_module module="Rscript src/arima_r.R [group]"`.
4. Finally you can evaluate the forecasts using `make run_module module="python -m src.evaluation"`.
