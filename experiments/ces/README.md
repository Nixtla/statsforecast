# CES experiments

## Main results

| dataset   | metric   |   AutoETS + AutoCES (StatsForecast) |   AutoETS (StatsForecast) |   AutoCES (StatsForecast) |   auto_ces (R) |
|:----------|:---------|-----------:|------:|------:|--------:|
| Yearly    | MASE     |      **2.992** | 3.338 | 3.417 |   3.35  |
| Yearly    | time     |      0.213 | 0.124 | 0.089 |   0.355 |
| Quarterly | MASE     |      **1.147** | 1.173 | 1.184 |   1.182 |
| Quarterly | time     |      2.935 | 1.441 | 1.494 |   2.081 |
| Monthly   | MASE     |      **0.927** | 0.97  | 0.936 |   0.943 |
| Monthly   | time     |     14.713 | 8.407 | 6.305 |   8.18  |
| Daily     | MASE     |      3.267 | 3.252 | 3.313 |   **3.251** |
| Daily     | time     |      1.217 | 0.428 | 0.789 |   1.01  |
| Weekly    | MASE     |      **2.392** | 2.405 | 2.464 |   2.426 |
| Weekly    | time     |      0.083 | 0.034 | 0.049 |   0.085 |
| Hourly    | MASE     |      1.273 | 1.608 | 1.184 |   **1.088** |
| Hourly    | time     |      1.284 | 0.298 | 0.986 |   0.21  |


(Time in minutes.)


## Reproducibility

To reproduce the main results you have:

1. Execute `make init`. 
2. Activate the environment using `conda activate ces`.
3. Run the experiments using `python -m src.ces --dataset M4 --group [group]` where `[model]` can be `[group]` can be `Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, and `Yearly`.
4. Compute the ensemble model using `python -m src.ensemble --dataset M4 --group [group]` where `[model]` can be `[group]` can be `Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, and `Yearly`.
4. To run R experiments you have to prepare the data using `python -m src.data --dataset M4 --group [group]` for each `[group]`. Once it is done, just run `make run_module module="Rscript src/ces_r.R [group]"`.
5. Finally you can evaluate the forecasts using `make run_module module="python -m src.evaluation"`.
