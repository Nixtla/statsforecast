# ARIMA experiments

## Main results

We compared accuracy and speed against [pmdarima](https://github.com/alkaline-ml/pmdarima), Rob Hyndman's [forecast](https://github.com/robjhyndman/forecast) package and Facebook's [Prophet](https://github.com/facebook/prophet). We used the `Daily`, `Hourly` and `Weekly` data from the [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128). 

The following table summarizes the results. As can be seen, our `auto_arima` is the best model in accuracy (measured by the `MASE` loss) and time, even compared with the original implementation in R.

| dataset | metric | auto_arima_nixtla | auto_arima_pmdarima [1] | auto_arima_r | prophet |
| :------ | :----- | ----------------: | ----------------------: | -----------: | ------: |
| Daily   | MASE   |          **3.26** |                    3.35 |         4.46 |   14.26 |
| Daily   | time   |          **1.41** |                   27.61 |         1.81 |  514.33 |
| Hourly  | MASE   |          **0.92** |                     --- |         1.02 |    1.78 |
| Hourly  | time   |         **12.92** |                     --- |        23.95 |   17.27 |
| Weekly  | MASE   |          **2.34** |                    2.47 |         2.58 |    7.29 |
| Weekly  | time   |              0.42 |                    2.92 |     **0.22** |   19.82 |


[1] The model `auto_arima` from `pmdarima` had a problem with Hourly data. An issue was opened.



The following table summarizes the data details.
 
| group  | n_series | mean_length | std_length | min_length | max_length |
| :----- | -------: | ----------: | ---------: | ---------: | ---------: |
| Daily  |    4,227 |       2,371 |      1,756 |        107 |      9,933 |
| Hourly |      414 |         901 |        127 |        748 |      1,008 |
| Weekly |      359 |       1,035 |        707 |         93 |      2,610 |

### ⏲ Computational efficiency

We measured the computational time against the number of time series. The following graph shows the results. As we can see, the fastest model is our `auto_arima`.

![](nbs/imgs/computational-efficiency.png)

<details>
    <summary> Nixtla vs Prophet </summary> 
    <img src="nbs/imgs/computational-efficiency-hours-wo-pmdarima.png" > 
</details>

You can reproduce the results [here](/experiments/arima/).

### External regressors

Results with external regressors are qualitatively similar to the ones reported before. You can find the complete experiments [here](/experiments/arima_xreg/).

## 👾 Less code
![pmd to stats](../../nbs/imgs/pdmarimaStats.gif)


## Reproducibility
To reproduce the main results, follow these steps:

1. **Initialize the Docker Image**: Execute make init to create a Docker image with all the necessary dependencies.

### Running testing one by one

2. **Run Python Experiments**: You can now run all Python experiments for a specific model or all of them at once:

- To run all `StatsForecast` experiments: make `run_tests_statsforecast`

- To run all `PMDARIMA` experiments: make `run_tests_pmdarima`

- To run all Prophet experiments: make run_tests_prophet

3. **Run R Experiments**:

    1. Prepare the data for R experiments by running: `make data_prep_r`
    2. Run all `R` experiments using: make `run_tests_r`

4. **Evaluate Forecasts**: Finally, evaluate the forecasts by executing: `make run_module module="python -m src.evaluation"`

### Additional Commands:

`Jupyter Lab`: To start a Jupyter Lab instance in the Docker container, run make jupyter. You can then access it in your browser at http://localhost:8888.

Bash in Docker: To open a bash shell inside the Docker container, use make bash_docker.
