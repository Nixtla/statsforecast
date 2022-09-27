# CES experiments

## Main results

| dataset   | metric   |   AutoCES (StatsForecast) |   auto_ces (R) | 
|:----------|:---------|------:|--------:|
| Daily     | MASE     | 3.313 |   3.251 |
| Daily     | time     | 0.789 |   1.01  |
| Hourly    | MASE     | 1.183 |   1.088 |
| Hourly    | time     | 1.054 |   0.21  |
| Monthly   | MASE     | 0.936 |   0.943 |
| Monthly   | time     | 6.305 |   8.18  |
| Quarterly | MASE     | 1.184 |   1.182 |
| Quarterly | time     | 1.494 |   2.081 |
| Weekly    | MASE     | 2.464 |   2.426 |
| Weekly    | time     | 0.049 |   0.085 |
| Yearly    | MASE     | 3.417 |   3.35  |
| Yearly    | time     | 0.089 |   0.355 |

(Time in minutes.)


## Reproducibility

To reproduce the main results you have:

1. Execute `make init`. 
2. Activate the environment using `conda activate ets`.
3. Run the experiments using `make run_module module="python -m src.ces --dataset M4 --group [group]` where `[model]` can be `[group]` can be `Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, and `Yearly`.
4. To run R experiments you have to prepare the data using `python -m src.data --dataset M4 --group [group]` for each `[group]`. Once it is done, just run `make run_module module="Rscript src/et_r.R [group]"`.
5. Finally you can evaluate the forecasts using `make run_module module="python -m src.evaluation"`.
