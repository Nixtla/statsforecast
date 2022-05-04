# ARIMA experiments

## M3


| frequency   | metric   | arima_prophet_adapter   |   prophet |
|:------------|:---------|------------------------:|----------:|
| Monthly     | mape     | **21.19**               |     22.37 |
| Monthly     | smape    | **15.17**               |     17.83 |
| Monthly     | time     | **1.23**                |     55.81 |
| Other       | mape     | **5.34**                |      6.39 |
| Other       | smape    | **4.87**                |      5.8  |
| Other       | time     | **0.31**                |      4.79 |
| Quarterly   | mape     | **13.99**               |     16.47 |
| Quarterly   | smape    | **10.57**               |     13.2  |
| Quarterly   | time     | **0.37**                |     38.49 |
| Yearly      | mape     | **22.33**               |     26.64 |
| Yearly      | smape    | **18.13**               |     21.04 |
| Yearly      | time     | **0.32**                |     45.68 |

## M4

| frequency   | metric   | arima_prophet_adapter   | prophet   |
|:------------|:---------|------------------------:|----------:|
| Daily       | mape     | **3.97**                | 11.32     |
| Daily       | smape    | **3.16**                | 9.05      |
| Daily       | time     | **3.15**                | 275.6     |
| Hourly      | mape     | **16.14**               | 26.4      |
| Hourly      | smape    | **13.58**               | 18.11     |
| Hourly      | time     | 27.76                   | **12.55** |
| Monthly     | mape     | **15.99**               | 20.79     |
| Monthly     | smape    | **13.71**               | 18.43     |
| Monthly     | time     | **80.78**               | 1238.44   |
| Quarterly   | mape     | **12.19**               | 18.45     |
| Quarterly   | smape    | **10.9**                | 15.5      |
| Quarterly   | time     | **4.81**                | 1097.17   |
| Weekly      | mape     | **8.23**                | 15.41     |
| Weekly      | smape    | **8.94**                | 17.3      |
| Weekly      | time     | **0.44**                | 10.83     |
| Yearly      | mape     | **17.77**               | 21.0      |
| Yearly      | smape    | **16.23**               | 17.0      |
| Yearly      | time     | **1.42**                | 1123.69   |


## Tourism

| frequency   | metric   | arima_prophet_adapter   | prophet   |
|:------------|:---------|------------------------:|----------:|
| Monthly     | mape     | **21.63**               | 25.39     |
| Monthly     | smape    | **20.63**               | 23.19     |
| Monthly     | time     | **3.64**                | 8.62      |
| Quarterly   | mape     | **16.66**               | 21.3      |
| Quarterly   | smape    | **16.43**               | 20.02     |
| Quarterly   | time     | **0.49**                | 21.39     |
| Yearly      | mape     | 29.31                   | **29.29** |
| Yearly      | smape    | 25.49                   | **24.64** |
| Yearly      | time     | **0.32**                | 21.5      |


## Reproducibility

Toc   | arima_prophet_adapter   |   prophet |
|:------------|:---------|:------------------------|----------:|
| Monthly     | mape     | **21.19**               |     22.37 |
| Monthly     | smape    | **15.17**               |     17.83 |
| Monthly     | time     | **1.23**                |     55.81 |
| Other       | mape     | **5.34**                |      6.39 |
| Other       | smape    | **4.87**                |      5.8  |
| Other       | time     | **0.31**                |      4.79 |
| Quarterly   | mape     | **13.99**               |     16.47 |
| Quarterly   | smape    | **10.57**               |     13.2  |
| Quarterly   | time     | **0.37**                |     38.49 |
| Yearly      | mape     | **22.33**               |     26.64 |
| Yearly      | smape    | **18.13**               |     21.04 |
| Yearly      | time     | **0.32**                |     45.68 | reproduce the main results you have:

1. Execute `make init` to create a Docker image with the required dependencies.
2. Run the experiments using `make run_module module="python -m src.[model] --dataset M4 --group [group]"` where `[model]` can be `statsforecast`, `pmdarima` and `prophet`, and `[group]` can be `Daily`, `Hourly` and `Weekly`.
3. To run R experiments you have to prepare the data using `make run_module module="python -m src.data --dataset M4 --group [group]"` for each `[group]`. Once it is done, just run `make run_module module="Rscript src/arima_r.R [group]"`.
4. Finally you can evaluate the forecasts using `make run_module module="python -m src.evaluation"`.
