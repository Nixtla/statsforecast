# ARIMA experiments

## Main Results

We compared accuracy and speed against [pmdarima](https://github.com/alkaline-ml/pmdarima), Rob Hyndman's [forecast](https://github.com/robjhyndman/forecast) package and Facebook's [Prophet](https://github.com/facebook/prophet). We used the `Daily`, `Hourly` and `Weekly` data from the [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128). 

The following table summarizes the results. The time results are relative to the `auto_arima_r` column, and the `MASE` loss values are shown as absolute values. As can be seen, our auto_arima is the most accurate model. These results were run in GitHub Codespaces on a 16-core machine.

| dataset | metric | auto_arima_nixtla | auto_arima_pmdarima | auto_arima_r |    prophet |
| :------ | :----- | ----------------: | ------------------: | -----------: | ---------: |
| Daily   | MASE   |          **3.25** |                3.35 |       218.62 |       14.7 |
| Daily   | time   |            1.5905 |             100.362 |        **1** |   147.4044 |
| Hourly  | MASE   |          **0.95** |                   - |        22973 |       1.79 |
| Hourly  | time   |          0.736783 |                   - |            1 | **0.1999** |
| Weekly  | MASE   |          **2.27** |                2.46 |       104.01 |       7.26 |
| Weekly  | time   |           1.81081 |             208.595 |        **1** |         63 |

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
