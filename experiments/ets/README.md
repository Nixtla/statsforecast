# ETS experiments

## Main results


| dataset   | metric   |   ets_statsforecast |   ets_r |   ets_statsmodels[1] |
|:----------|:---------|-------------:|--------:|------------------:|
| Hourly    | MASE     |         1.61 |    1.82 |   21848.5         |
| Hourly    | time     |        18.79 |   35.45 |     112.35        |
| Daily     | MASE     |         3.23 |    3.25 |     325.42        |
| Daily     | time     |        26.24 |   17.78 |      19.97        |
| Weekly    | MASE     |         2.55 |    2.53 |       2.68        |
| Weekly    | time     |         1.78 |    2.12 |       1.56        |
| Monthly   | MASE     |         0.97 |    0.95 |       3.75016e+07 |
| Monthly   | time     |       512.7  |  907.23 |    2285.11        |
| Quarterly | MASE     |         1.17 |    1.16 |       9.01169e+07 |
| Quarterly | time     |        88.48 |   75.78 |     280.89        |
| Yearly    | MASE     |         3.09 |    3.44 |     101.64        |
| Yearly    | time     |         6.73 |   15.38 |      34.35        |
[1] The model `ETSModel` from `statsmodels` had performance problems for particular series. An [issue](https://github.com/statsmodels/statsmodels/issues/8344) was opened and answered.

## Reproducibility

To reproduce the main results you have:

1. Execute `conda env create -f environment.yml`. 
2. Activate the environment using `conda activate ets`.
3. Run the experiments using `python -m src.[model] --dataset M4 --group [group]` where `[model]` can be `statsforecast`, and `[group]` can be `Daily`, `Hourly` and `Weekly`.
4. To run R experiments you have to prepare the data using `python -m src.data --dataset M4 --group [group]` for each `[group]`. Once it is done, just run `Rscript src/ets_r.R [group]`.
5. Finally you can evaluate the forecasts using `python -m src.evaluation`.
