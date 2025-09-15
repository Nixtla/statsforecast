# ARIMA experiments

## Main Results

We compared accuracy and speed against [pmdarima](https://github.com/alkaline-ml/pmdarima), Rob Hyndman's [forecast](https://github.com/robjhyndman/forecast) package and Facebook's [Prophet](https://github.com/facebook/prophet). We used the `Daily`, `Hourly` and `Weekly` data from the [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128). 

The following table summarizes the results. This table is relative to the performanace of auto_arima_r column. As can be seen, our `auto_arima` is the best model in accuracy (measured by the `MASE` loss) and time, even compared with the original implementation in R. This results were run in Github Codespaces with a 16 core machine.

| dataset | metric | auto_arima_nixtla | auto_arima_pmdarima | auto_arima_r | prophet |
| :------ | :----- | ----------------: | ------------------: | -----------: | ------: |
| Daily   | MASE   |      **0.014866** |           0.0153234 |            1 |       - |
| Daily   | time   |           1.57072 |             100.362 |        **1** |       - |
| Hourly  | MASE   |   **4.13528e-05** |                   - |            1 |    1.79 |
| Hourly  | time   |      **0.736783** |                   - |            1 |   16.04 |
| Weekly  | MASE   |     **0.0218248** |           0.0236516 |            1 |    7.26 |
| Weekly  | time   |           1.83784 |             208.595 |        **1** |   23.31 |

The following table is the raw results in seconds for time and MASE. 

 | dataset | metric | auto_arima_nixtla | auto_arima_pmdarima | auto_arima_r | prophet |
 | :------ | :----- | ----------------: | ------------------: | -----------: | ------: |
 | Daily   | MASE   |          **3.25** |                3.35 |       218.62 |       - |
 | Daily   | time   |              6.33 |              404.46 |     **4.03** |       - |
 | Hourly  | MASE   |          **0.95** |                   - |        22973 |    1.79 |
 | Hourly  | time   |         **59.09** |                   - |         80.2 |   16.04 |
 | Weekly  | MASE   |          **2.27** |                2.46 |       104.01 |    7.26 |
 | Weekly  | time   |              0.68 |               77.18 |     **0.37** |   23.31 |



## Reproducibility
To reproduce the main results, follow these steps:

1. **Initialize the Docker Image**: Execute make init to create a Docker image with all the necessary dependencies.


2. **Run Python Experiments**: You can now run all Python experiments for a specific model or all of them at once:

- To run all `StatsForecast` experiments: `make run_tests_statsforecast`
- To run all `PMDARIMA` experiments: `make run_tests_pmdarima`
- To run all `Prophet` experiments: `make run_tests_prophet`

3. **Run R Experiments**:

    1. Prepare the data for R experiments by running: `make data_prep_r`
    2. Run all `R` experiments using: make `make run_tests_r`

4. **Evaluate Forecasts**: Finally, evaluate the forecasts by executing: `make run_evaluation`

### Running trials for debugging 
- `StatsForecast`:
`make run_module module="python -m src.statsforecast --dataset M3Small --group Yearly"`
- `PMDARIMA`:
`make run_module module="python -m src.pmdarima --dataset M3Small --group Yearly"`
- `Prophet`:
`make run_module module="python -m src.prophet --dataset M3Small --group Yearly"`
- `R`:
`make run_module module="python -m src.data --dataset M3Small --group Yearly"`
`make run_module module="python -m src/arima_forecast_r.R  --dataset M3Small --group Yearly"`
`make run_module module="python -m src/arima_fable_r.R --dataset M3Small --group Yearly"`

### Additional Commands:

`Jupyter Lab`: To start a Jupyter Lab instance in the Docker container, run make jupyter. You can then access it in your browser at http://localhost:8888.

Bash in Docker: To open a bash shell inside the Docker container, use make bash_docker.
