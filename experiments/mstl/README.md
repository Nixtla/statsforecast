# MSTL is faster and more accurate than Prophet and NeuralProphet with multiple seasonalities

Some time series are generated from very low frequency data. These data generally exhibit multiple seasonalities. For example, hourly data may exhibit repeated patterns every hour (every 24 observations) or every day (every 24 * 7, hours per day, observations). This is the case for electricity load. Electricity load may vary hourly, e.g., during the evenings electricity consumption may be expected to increase. But also, the electricity load varies by week. Perhaps on weekends there is an increase in electrical activity.

## MSTL model

The `MSTL` (Multiple Seasonal-Trend decomposition using LOESS) model, originally developed by [Kasun Bandara, Rob J Hyndman and Christoph Bergmeir](https://arxiv.org/abs/2107.13462), decomposes the time series in multiple seasonalities using a Local Polynomial Regression (LOESS). Then it forecasts the trend using a custom non-seasonal model and each seasonality using a `SeasonalNaive` model.

`StatsForecast` contains a fast implementation of the `MSTL` model. Also, the decomposition of the time series can be calculated.

## Install StatsForecast
```bash
pip install statsforecast
```

## Experiment

To validate the accuracy of the `MSTL` model, we will show its performance on unseen data. We will use a classical time series technique called cross validation. Cross-Validation is a widely used technique in data science and machine learning. Most common cross-validation methods require shuffling of the data, and this is why those methods are not applicable to time-series data. The nature of time-series data requires a distinct approach to cross-validation.

In this procedure there is a series of test sets. The corresponding training sets consist only of observations that ocurred prior to the observations from the test set. Thus, no future observations can be used in constructing the forecast. The following diagram illustrates the series of training and test sets, where the blue observations form the training sets, and the orange observations form the test sets. The forecast accuracy is computed by averaging over the test sets.

![image](https://user-images.githubusercontent.com/10517170/203424908-e53bfa68-3740-45bf-a693-8119d9cdda94.png)


### Main Results

In this experiment we used the PJM dataset:

> PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system serving all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia. The hourly power consumption data comes from PJM’s website and are in megawatts (MW).

#### Accuracy

The performance of the models was evaluated retraining the models each 24 hours and forecasting the next 24 hours for 7 days.

<img width="528" alt="image" src="https://user-images.githubusercontent.com/10517170/203425099-c096d129-d26e-4af9-a048-2b31549e9bfc.png">

#### Time

| Model | Time (mins) |
| -------| -----------|                                  
| MSTL   |  1.066439|                                    
|Prophet  |2.343852|                            
|  NeuralProphet  |9.643826 |


### Reproducibility


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


