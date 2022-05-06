# AutoARIMAProphet experiments

[Prophet](https://github.com/facebook/prophet) is one of the most widely used time series forecasting models in the world. Its GitHub repository has more than 14 thousand stars and more than a hundred repositories depending on the implementation. However, in many scenarios, [it does not offer good performance in terms of time and accuracy](https://analyticsindiamag.com/why-are-people-bashing-facebook-prophet/). This is highly relevant when you want to forecast thousands of time series. The success of Prophet depends to a large extent on its usability, for example, [adding exogenous and calendar variables is almost trivial](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html). For this purpose, we have created a Prophet API adapter to use Prophet's functionalities but with a faster and more accurate model such as AutoARIMA. Just import this adapter and replace it with the Prophet class to start using AutoARIMA inside a pipeline made for Prophet.

```python
from prophet import Prophet
from statsforecast.adapters.prophet import AutoARIMAProphet

# BEFORE
m = Prophet()
# AFTER
m = AutoARIMAProphet()
m.fit(df)
future = m.make_future_dataframe(365)
forecast = m.predict(future)
fig = m.plot(forecast)
```

With this simple change, a **reduction of 17% in MAPE, 15% in SMAPE can be achieved**. Also, `AutoARIMAProphet` is **31x** faster.

# Experiment details

To validate the Prophet adapter, we design a pipeline considering the M3, M4, and Tourism datasets, widely used to create benchmarks. The pipeline finds the best hyperparameters of Prophet by doing cross-validation. We simply replace Prophet with AutoARIMAProphet in the pipeline without performing hyperparameter optimization (AutoARIMA performs it inside the model).

# Results 

The following tables show the results for each dataset (time in minutes). 

![comparison](./comparison.png)

## M3

| frequency   | metric   | arima_prophet_adapter   |   prophet | %_reduction_vs_prophet   |
|:------------|:---------|------------------------:|----------:|-------------------------:|
| Monthly     | mape     | **21.19**               |     22.37 | -5.27%                   |
| Monthly     | smape    | **15.17**               |     17.83 | -14.92%                  |
| Monthly     | time     | **1.23**                |     55.81 | -97.80%                  |
| Other       | mape     | **5.34**                |      6.39 | -16.43%                  |
| Other       | smape    | **4.87**                |      5.8  | -16.03%                  |
| Other       | time     | **0.31**                |      4.79 | -93.53%                  |
| Quarterly   | mape     | **13.99**               |     16.47 | -15.06%                  |
| Quarterly   | smape    | **10.57**               |     13.2  | -19.92%                  |
| Quarterly   | time     | **0.37**                |     38.49 | -99.04%                  |
| Yearly      | mape     | **22.33**               |     26.64 | -16.18%                  |
| Yearly      | smape    | **18.13**               |     21.04 | -13.83%                  |
| Yearly      | time     | **0.32**                |     45.68 | -99.30%                  |

## M4

| frequency   | metric   | arima_prophet_adapter   | prophet   | %_reduction_vs_prophet   |
|:------------|:---------|------------------------:|----------:|-------------------------:|
| Daily       | mape     | **3.97**                | 11.32     | -64.93%                  |
| Daily       | smape    | **3.16**                | 9.05      | -65.08%                  |
| Daily       | time     | **3.15**                | 275.6     | -98.86%                  |
| Hourly      | mape     | **16.14**               | 26.4      | -38.86%                  |
| Hourly      | smape    | **13.58**               | 18.11     | -25.01%                  |
| Hourly      | time     | 27.76                   | **12.55** | 121.20%                  |
| Monthly     | mape     | **15.99**               | 20.79     | -23.09%                  |
| Monthly     | smape    | **13.71**               | 18.43     | -25.61%                  |
| Monthly     | time     | **80.78**               | 1238.44   | -93.48%                  |
| Quarterly   | mape     | **12.19**               | 18.45     | -33.93%                  |
| Quarterly   | smape    | **10.9**                | 15.5      | -29.68%                  |
| Quarterly   | time     | **4.81**                | 1097.17   | -99.56%                  |
| Weekly      | mape     | **8.23**                | 15.41     | -46.59%                  |
| Weekly      | smape    | **8.94**                | 17.3      | -48.32%                  |
| Weekly      | time     | **0.44**                | 10.83     | -95.94%                  |
| Yearly      | mape     | **17.77**               | 21.0      | -15.38%                  |
| Yearly      | smape    | **16.23**               | 17.0      | -4.53%                   |
| Yearly      | time     | **1.42**                | 1123.69   | -99.87%                  |


## Tourism

| frequency   | metric   | arima_prophet_adapter   | prophet   | %_reduction_vs_prophet   |
|:------------|:---------|------------------------:|----------:|-------------------------:|
| Monthly     | mape     | **21.63**               | 25.39     | -14.81%                  |
| Monthly     | smape    | **20.63**               | 23.19     | -11.04%                  |
| Monthly     | time     | **3.64**                | 8.62      | -57.77%                  |
| Quarterly   | mape     | **16.66**               | 21.3      | -21.78%                  |
| Quarterly   | smape    | **16.43**               | 20.02     | -17.93%                  |
| Quarterly   | time     | **0.49**                | 21.39     | -97.71%                  |
| Yearly      | mape     | 29.31                   | **29.29** | 0.07%                    |
| Yearly      | smape    | 25.49                   | **24.64** | 3.45%                    |
| Yearly      | time     | **0.32**                | 21.5      | -98.51%                  |


## Reproducibility


1. Create a conda environment using the `environment.yml` (`conda env create -f environment.yml`).
2. Activate the conda environment using `conda activate arima_prophet`.
3. Run the experiments for each dataset and each model using `python -m src.experiment --dataset [dataset] --group [group] --model_name [model_name]`. For `M4`, the groups are `Yearly`, `Monthly`, `Quarterly`, `Weekly`, `Daily`, and `Hourly`. For `M3`, the groups are `Yearly`, `Monthly`, `Quarterly`, and `Other`. For `Tourism`, the groups are `Yearly`, `Monthly`, and `Quarterly`. Finally, for `PeytonManning` the group is `Daily`.
4. Evaluate the results using `python -m src.evaluation`.
