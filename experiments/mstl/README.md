# Seasonal Naive is faster and more accurate than Prophet and NeuralProphet for multiple seasonalities

## TL;DR

For next-day forecasting of electricity load:

* A Seasonal Naive model is over 20% more accurate and a lot faster than Prophet and Neural Prophet 
* A simple MSTL model is, in comparison, 5% more accurate than a seasonal naive. 

## Intro

Some time series are generated from very low-frequency data. This data generally exhibits multiple seasonalities. For example, hourly data may exhibit repeated patterns every day (24H) or every week (168H). 

Electricity load data is such an example. Seasonalities can be observed at some hours of the day (e.g. evenings vs mornings) or on certain days of the week (Mondays vs Sundays)


In this experiment, We will use historic hourly electricity load data from PJM to forecast the next 24 hours. PJM is a regional transmission organization (RTO) that coordinates the movement of wholesale electricity in all or parts of 13 states and the District of Columbia. The original data can be found [here](https://github.com/jnagura/Energy-consumption-prediction-analysis).

We will compare Prophet and NeuralProphet against a Seasonal Naive and an MSTL model for this setting. 


## Experiment

### Data

In this experiment, we used the PJM dataset. PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system serving all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia. The hourly power consumption data comes from PJM’s website and are in megawatts (MW).

The dataset contains a unique series with seasonal patterns and 32,896 observations.

### Models

#### Prophet

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
[Documentation](https://facebook.github.io/prophet/)


#### NeuralProphet

Based on Neural Networks, inspired by Facebook Prophet and AR-Net, built on Pytorch.
[Documentation](https://neuralprophet.com/contents.html)

#### Seasonal Naive

The `SeasonalNaive` model is a simple baseline model that sets each forecast to be equal to the last observed value from the same season (e.g., the same hour of the previous day).

`StatsForecast` contains a fast implementation of the `SeasonalNaive` model. 

#### MSTL model

The `MSTL` (Multiple Seasonal-Trend decomposition using LOESS) model, originally developed by [Kasun Bandara, Rob J Hyndman and Christoph Bergmeir](https://arxiv.org/abs/2107.13462), decomposes the time series in multiple seasonalities using a Local Polynomial Regression (LOESS). Then it forecasts the trend using a custom non-seasonal model and each seasonality using a `SeasonalNaive` model.

`StatsForecast` contains a fast implementation of the `MSTL` model.
[Documentation](https://nixtla.github.io/statsforecast/examples/multipleseasonalities.html)


### Performance Evaluation

We use cross validation to evaluate the performance or accuracy of the models. We report the following error metrics: Mean Absolute Standard Error (MASE), Mean Absolute Error (MAE), Symmetric Mean Absolute Percent Error (SMAPE) and  Root Mean Squared Error.

With time series data, cross validation is done by defining a sliding window across the historical data and predicting the period following it. This form of cross-validation allows us to arrive at a better estimation of our model’s predictive abilities across a wider range of temporal instances while also keeping the data in the training set contiguous as is required by our models.

Due to the sequential nature of time, common cross-validation methods that require the shuffling of the data are not applicable.

The following diagram illustrates the series of training and test sets, where the blue observations form the training sets, and the orange observations form the test sets. The forecast accuracy is computed by averaging over the test sets.

![image](https://user-images.githubusercontent.com/10517170/203424908-e53bfa68-3740-45bf-a693-8119d9cdda94.png)

We re-training the models every 24 hours and forecasted the next 24 hours for 7 days using this method.

## Results

For this setting: 

* SeasonalNaive is 20% more accurate than NeuralProphet and 366 times faster. StatsForecast's MSTL is 24% more accurate than NeuralProphet and 11 times faster.

* SeasonalNaive is 23% more accurate than Prophet and 75 times faster. StatsForecast's MSTL is 26% more accurate than Prophet and 2 times faster.


### Accuracy

The performance of the models was evaluated re-training the models every 24 hours and forecasting the next 24 hours for 7 days using cross-validation for time series.

<img width="528" alt="image" src="https://user-images.githubusercontent.com/10517170/203425099-c096d129-d26e-4af9-a048-2b31549e9bfc.png">

### Time

| Model | Time (mins) |
| -------| -----------|
| SN | 0.03 |                                  
| MSTL   |  1.066439|                                    
|Prophet  |2.343852|                            
|  NeuralProphet  |9.643826 |

## Conclusion

For this dataset and specific setting, simpler models outperform fancy and complex models both in accuracy and speed. Sometimes, simpler is better. So choose your forecasting models carefully and according to the task at hand.

## Disclaimer
These results don't imply that you should always prefer MSTL over other models. For example, for long-horizon tasks (e.g. forecast next week) NeuralProphet or NHiTS models might be better suited. 


## Reproducibility


1. Create a conda environment `exp_mstl` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate exp_mstl
  ```

4. Run the experiments for each dataset and each model using 
  ```shell
  python -m src.main
  ```


## Misc.

* [`StatsForecast`](https://github.com/nixtla/statsforecast) also includes a variety of lightning fast baseline models.
* If you really need to do forecast at scale, [here](https://github.com/nixtla/statsforecast/tree/main/experiments/ray) we show how to forecast 1 million time series under 30 minutes using [Ray](https://github.com/ray-project/ray).
* If you are interested in SOTA Deep Learning models, check [`NeuralForecast`](https://github.com/nixtla/neuralforecast)


