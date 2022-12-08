## Amazon Forecast vs open source statistical methods
>TL;DR: We paid USD $1,404.00 and spend 4 hours in the AWS console so you don't have to. 

In this reproducible experiment, we compare [Amazon Forecast](https://aws.amazon.com/forecast/) and the open-source library [StatsForecast](https://github.com/Nixtla/statsforecast). Given the prominent use of AWS Forecast in demand forecasting, for this experiment, we used the 30,490 series of daily sales at Walmart from the [M5 competition](https://mofc.unic.ac.cy/m5-competition/). This dataset is interesting for its scale but also the fact that it features many timeseries with infrequent occurances.  Such timeseries are common in retail scenarios and are difficult for traditional timeseries forecasting techniques to address. 


### Amazon Forecast
Amazon Forecast is a time-series forecasting service based on machine learning (ML) and built for business metrics analysis. 

> It uses machine learning (ML) to generate more accurate demand forecasts with just a few clicks, without requiring any prior ML experience. Amazon Forecast includes algorithms that are based on over twenty years of forecasting experience and developed expertise used by Amazon.com bringing the same technology used at Amazon to developers as a fully managed service, removing the need to manage resources. Amazon Forecast uses ML to learn not only the best algorithm for each item, but the best ensemble of algorithms for each item, automatically creating the best model for your data.

Amazon Forecast is one of the leading forecasting services out there. You can read more about its features and pricing tiers here. 

### StatsForecast
StatsForecast is an open-source python library from Nixtla. The library offers a collection of widely used univariate time series forecasting models, including automatic ARIMA, ETS, CES, and Theta modeling optimized for high performance using numba. It also includes a large battery of benchmarking models.

### Main Results

Amazon Forecast achieved 1.617 in error (measured in wRMSSE, the official evaluation metric used in the competition), took 4.1 hours to run and cost 1,404 USD. Statsforecast, on a `c5d.24xlarge` EC2 instance, achieved 0.669 in error (wRMSSE), took 14.5 minutes to run and cost only 1.2 USD. 

For this data set, we show therefore that: 

* Amazon Forecast is 60% less accurate and 1,100 times more expensive than running an open-source alternative in a cloud server. 
* Machine Learning methods are outperformed by classical methods in terms of speed, accuracy and cost. 

Although using StatsForecast requires some basic knowledge of Python and cloud computing, the results are simply better. 

## Data

The data are ready for download at the following URLs:

- Train set: 
    `https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet`
- Temporal exogenous variables (used by AmazonForecast): 
    `https://m5-benchmarks.s3.amazonaws.com/data/train/temporal.parquet`
- Static exogenous variables (used by AmazonForecast): 
    `https://m5-benchmarks.s3.amazonaws.com/data/train/static.parquet`
    
Tha train set contains `30,490` time series. The M5 competition is hierarchical. That is, forecasts are required for different levels of aggregation: national, state, store, etc. In this experiment, we only generate forecasts using the bottom-level data. The evaluation is performed using the bottom-up reconciliation method to obtain the forecasts for the higher hierarchies. 

## Experiment

The experiment consists of forecasting 28 days forward for 30,490 time series. The series correspond to the bottom level of the M5 competition. The evaluation is performed hierarchically. That is, the forecasts are aggregated to obtain the higher levels and subsequently evaluate the accuracy of each one of them.

The evaluation metric is the weighted Root Mean Scaled Squared Error (wRMSSE) defined as follows,

<img width="237" alt="image" src="https://user-images.githubusercontent.com/10517170/206323185-cad22076-cf91-40ea-a060-a83704bf790a.png">

Where the `RMSSE` is defined by,

<img width="648" alt="image" src="https://user-images.githubusercontent.com/10517170/206323251-7e2a3318-1f24-4f34-ad32-d872464454b4.png">

The `wRMSSE` is the official metric used in the M5 competition.

We ran the Amazon forecast according to the the instructions and forecasted 3 quantiles for the next 28 days.
We spined of a `c5d.24xlarge` EC2 instance and run StatsForecast with just 4 models: `AutoETS`, `DynamicOptimizedTheta` and `SeasonalNaive`. Finally, the `AutoETS` and `DynamicOptimizedTheta` models were ensembled using the median. 

Detailed results per dataset are shown below.

## Results

### Performance

The table shows the performance of `StatsForecast` and `AmazonForecast` on the M5 dataset using the official evaluation metric.

<img width="652" alt="image" src="https://user-images.githubusercontent.com/10517170/206319955-5c1e9a05-811e-4113-84f0-30df6ed86a19.png">

The following table shows the performance across all levels of the hierarchy:

<img width="652" alt="image" src="https://user-images.githubusercontent.com/10517170/206324561-2d4af50b-ca92-44df-ad80-fa970c960c43.png">

### Time

Excluding time spent in the console and just accounting for processing and computing time, Amazon Forecast took 4.1 hours to run. In comparison, StatsForecast took just 15 minutes to run. Running time includes the end-to-end pipeline including loading data, training and forecasting. 

### Cost 

Amazon included a cost calculator that is quite accurate. We fill in the followig data corresponding to the experiment. XXX
The estimated cost was 1,404 USD. 

In comparison, we paid 1.2 USD of EC2 associated costs. (This could have been further reduced by using spot instances.)

Below, you can find the detailed results. 

**Speed and Cost on M5 Data set**

| Model | Time (hours) | Cost (USD) |
|:-------|-------------:|-----:|
| StatsForecast | 0.26 |  1.2    |
| AmazonForecast | 4.1 | 1,404.05 |


## Conclusions

Although this experiment does not focus on comparing machine learning and deep learning vs statistical methods, it supports our [previous conclusions](https://github.com/Nixtla/statsforecast/tree/main/experiments/m3) on the current validity of simpler methods for many forecasting tasks.

## Unsolicited Advice

Don't always grab for the low-hanging fruit. It pays off. 

## Reproducibility
You can fully reproduce the experiment by following [this step-by-step notebook](https://nixtla.github.io/statsforecast/examples/aws/statsforecast.html).
