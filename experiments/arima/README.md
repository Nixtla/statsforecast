# ARIMA experiments

## Reproducibility
To reproduce the main results, follow these steps:

1. **Initialize the Docker Image**: Execute make init to create a Docker image with all the necessary dependencies.


2. **Run Python Experiments**: You can now run all Python experiments for a specific model or all of them at once:

- To run all `StatsForecast` experiments: `make run_tests_statsforecast`
- To run all `PMDARIMA` experiments: `make run_tests_pmdarima`
- To run all `Prophet` experiments: `make run_tests_prophet`

3. **Run R Experiments**:

    1. Prepare the data for R experiments by running: `make data_prep_r`
    2. Run all `R` experiments using: make `run_tests_r`

4. **Evaluate Forecasts**: Finally, evaluate the forecasts by executing: `make run_module module="python -m src.evaluation"`

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
