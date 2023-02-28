## Scalable Forecasting with Millions of Time-Series: Comparing Google's BigQuery with Open-Source Tools
>TL;DR: In this reproducible experiment, we compare BigQuery ML's forecasting solution with two open-source tools, StatsForecast and Fugue. The experiment concludes that **BigQuery is 13% less accurate, 8 times slower, and 10 times more expensive** than running an open-source alternative in a simple cloud cluster.


In this reproducible experiment, we compare [BigQuery's forecasting solution](https://cloud.google.com/bigquery-ml/docs/arima-speed-up-tutorial) with two open source tools: [StatsForecast](https://github.com/Nixtla/statsforecast) and [fugue](https://github.com/fugue-project/fugue). 

For this experiment, we followed [this experiment](https://cloud.google.com/bigquery-ml/docs/arima-speed-up-tutorial) used by Google to showcase their forecasting capabilities:

> For all steps but the last one, you will use the new_york.citibike_trips data. This data contains information about Citi Bike trips in New York City. This dataset only contains a few hundred time series. It is used to illustrate various strategies to accelerate model training. For the last step, you will use iowa_liquor_sales.sales data to forecast more than 1 million time series.

We conclude that, for this setting, **BigQuery is 13% less accurate, 8 times slower, and 10 times more expensive** than running an open-source alternative in a simple cloud cluster.

### Google's BigQuery

BigQuery inclues an AutoML time-series forecasting service. 

> The BigQuery ML time series modeling pipeline includes multiple modules. The ARIMA model is the most computationally expensive, which is why the model is named ARIMA_PLUS. The modeling pipeline for the BigQuery ML time series includes the following functionalities: Infer the data frequency of the time series. Handle irregular time intervals. Handle duplicated timestamps by taking the mean value. Interpolate missing data using local linear interpolation. Detect and clean spike and dip outliers. Detect and adjust abrupt step (level) changes. Detect and adjust holiday effect. Detect multiple seasonal patterns within a single time series via Seasonal and Trend decomposition using Loess (STL), and extrapolate seasonality via double exponential smoothing (ETS). Detect and model the trend using the ARIMA model and the auto.ARIMA algorithm for automatic hyperparameter tuning. In auto.ARIMA, dozens of candidate models are trained and evaluated in parallel. The best model comes with the lowest Akaike information criterion (AIC).

With BigQuery ML, you can easily create and execute machine learning models in BigQuery using Google SQL queries. This tool also comes with an automated forecasting feature that can produce forecasts effortlessly. Moreover, BigQuery uses statistical models like AutoARIMA to model trends and seasonality, ensuring accurate forecasting.

### Fugue 

Fugue is a unified interface for distributed computing that lets users execute Python, Pandas, and SQL code on Spark, Dask, and Ray with minimal rewrites. The librarie also has a native [BigQuery integration](https://fugue-tutorials.readthedocs.io/tutorials/integrations/warehouses/bigquery.html?highlight=bigquery#the-bigquery-client).  

### StatsForecast

StatsForecast is an open-source python library from Nixtla. The library offers a collection of widely used univariate time series forecasting models, including automatic ARIMA, ETS, CES, and Theta modeling optimized for high performance using numba. It also includes a large battery of benchmarking models.

For this experiment, we used a databricks cluster of 16 e2-standard-32 virtual machines (GCP) to train five simple statistical models: `MSTL`, `AutoETS`, `AutoCES`, `Naive`, and `SeasonalNaive`.

### Main Results

Google's BigQuery:

* Achieved 24.13 (Mean Absolute Error, MAE) in error for the new_york.citibike_trips dataset.
* Took 7.5 minutes to run the new_york.citibike_trips dataset (approximately 400 time series).
* Took 1 hour 16 minutes to run the iowa_liquor_sales.sales dataset (over a million time series).
* Cost 41.96 USD.

StatsForecast and Fugue trained on a databricks cluster of 16 e2-standard-32 virtual machines (GCP):

* Achieved 20.96 (Mean Absolute Error, MAE) in error for the new_york.citibike_trips dataset.
* Took 2 minutes to run the new_york.citibike_trips dataset (approximately 400 time series).
* Took 9 minutes to run the iowa_liquor_sales.sales dataset (over a million time series).
* Cost only 4.02 USD.

Therefore, we show that for this dataset:

BigQuery is 13% less accurate, 8 times slower, and 10 times more expensive than running an open-source alternative in a simple cloud cluster.
Classical methods outperform complex methods and pipelines in terms of speed, accuracy, and cost.

Although using StatsForecast requires some basic knowledge of Python and cloud computing, the results are simply better.

## Experiment

### Citibike Trips (~400 time series)

The experiment consists of forecasting 7 days forward for approximately 400 time series. The largest bike share program in the USA is Citi Bike, spanning over Manhattan, Brooklyn, Queens, and Jersey City, with 10,000 bikes and 600 stations. This dataset encompasses Citi Bike trips from September 2013 when the program first launched and is regularly updated on a daily basis.

### Evaluation Metrics

The evaluation metrics are the Mean Absolute Error (MAE) and the Root Mean Squared Error (RMSE) defined as follows,

<img width="246" alt="image" src="https://user-images.githubusercontent.com/10517170/221758064-95101dbf-5d6e-4678-bbe8-e53d4f5a949f.png">

### Results

<img width="614" alt="image" src="https://user-images.githubusercontent.com/10517170/221756548-877c712e-95cd-48ae-96c6-e6648b6ce6a0.png">

###  Liquor Sales (~1 million time series)

According to the [bigquery's page](https://console.cloud.google.com/marketplace/details/iowa-department-of-commerce/iowa-liquor-sales?filter=category:machine-learning&id=18f0a495-8e20-4124-a349-0c4c167b60ab&project=fuguedatabricks):

> This dataset contains every wholesale purchase of liquor in the State of Iowa by retailers for sale to individuals since January 1, 2012. The State of Iowa controls the wholesale distribution of liquor intended for retail sale, which means this dataset offers a complete view of retail liquor sales in the entire state. The dataset contains every wholesale order of liquor by all grocery stores, liquor stores, convenience stores, etc., with details about the store and location, the exact liquor brand and size, and the number of bottles ordered.

### Results

<img width="507" alt="image" src="https://user-images.githubusercontent.com/10517170/221756599-fcdcfa7d-d1c3-405c-abd5-72f5457664c9.png">

## Conclusions

In conclusion, for the specific setting of this experiment, open source solutions outperformed Google's AutoML in terms of speed, costs,  simplicity, and accuracy. It should be noted, however, that this conclusion may not necessarily hold true for other datasets. Nonetheless, considering the uncertainty of the benefits and the certainty of cost and time, AutoML methods should be taken with a grain of salt.

## Unsolicited Advice
Choose your models wisely.

It would be extremely expensive and irresponsible to favor AutoML in an organization before establishing solid baselines.

Simpler is sometimes better. Not everything that glows is gold.

Go and try other great open-source libraries like GluonTS, Darts and Sktime.

## Reproducibility

You can fully reproduce the experiment by following [this step-by-step notebooks]:
- [Citibike Trips](./src/statsforecast-fugue-citibikes-trips.ipynb)
- [Liquor sales](./src/statsforecast-fugue-liquor-sales.ipynb)
