## Amazon's AutoML vs open source statistical methods
>TL;DR: We paid USD $800 USD and spend 4 hours in the AWS Forecast console so you don't have to. 

In this reproducible experiment, we compare [Amazon Forecast](https://aws.amazon.com/forecast/) and [StatsForecast](https://github.com/Nixtla/statsforecast) a python open-source library. For this experiment, given the prominent use of AWS Forecast in demand forecasting, we used the 30,490 series of daily sales at Walmart from the [M5 competition](https://mofc.unic.ac.cy/m5-competition/). We conclude that, for this setting,  Amazon Forecast is 60% less accurate and 669 times more expensive than running an open-source alternative in a simple cloud server. 

We also provide a step-by-step guide to [reproduce the results](https://nixtla.github.io/statsforecast/examples/aws/statsforecast.html).


### Amazon Forecast
Amazon Forecast is an AutoML time-series forecasting service. 

> It uses machine learning (ML) to generate more accurate demand forecasts with just a few clicks, without requiring any prior ML experience. Amazon Forecast includes algorithms that are based on over twenty years of forecasting experience and developed expertise used by Amazon.com bringing the same technology used at Amazon to developers as a fully managed service, removing the need to manage resources. Amazon Forecast uses ML to learn not only the best algorithm for each item, but the best ensemble of algorithms for each item, automatically creating the best model for your data.

Amazon Forecast is one of the leading forecasting services out there. You can read more about its features and pricing tiers [here](https://aws.amazon.com/forecast/).

Amazon Forecast creates predictors with AutoPredictor, which involves applying the optimal combination of algorithms to each time series in your datasets.  The predictor is an Amazon Forecast model that is trained using your target time series, related time series, item metadata, and any additional datasets you include. 

Included algorithms range from commonly used statistical algorithms like Autoregressive Integrated Moving Average (ARIMA), to complex neural network algorithms like CNN-QR and DeepAR+.: CNN-QR, DeepAR+, Prophet, NPTS, ARIMA, and ETS.


### StatsForecast

StatsForecast is an open-source python library from Nixtla. The library offers a collection of widely used univariate time series forecasting models, including automatic ARIMA, ETS, CES, and Theta modeling optimized for high performance using numba. It also includes a large battery of benchmarking models.

For this experiment, we used a `c5d.24xlarge` EC2 instance and trained two simple statistical models: `AutoETS`, and `DynamicOptimizedTheta`. Finally, the `AutoETS` and `DynamicOptimizedTheta` models were ensembled using the median. 


### Main Results

Amazon Forecast: 

* achieved 1.617 in error (measured in wRMSSE, the official evaluation metric used in the competition), 
* took 4.1 hours to run,
* and cost 803.53 USD. 

Statsforecast with a simple ensemble of statistical methods trained on a `c5d.24xlarge` EC2 instance:
* achieved 0.669 in error (wRMSSE), 
* took 14.5 minutes to run,
* and cost only 1.2 USD. 

For this data set, we show therefore that: 

* Amazon Forecast is 60% less accurate and 669 times more expensive than running an open-source alternative in a simple cloud server. 
* Machine Learning methods are outperformed by classical methods in terms of speed, accuracy and cost. 

Although using StatsForecast requires some basic knowledge of Python and cloud computing, the results are simply better for this dataset.

## Data

We provide open access to the data in the following URLs:

- Train set: 
    `https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet`
- Temporal exogenous variables (used by AmazonForecast): 
    `https://m5-benchmarks.s3.amazonaws.com/data/train/temporal.parquet`
- Static exogenous variables (used by AmazonForecast): 
    `https://m5-benchmarks.s3.amazonaws.com/data/train/static.parquet`
    
The train set contains `30,490` time series. The M5 competition is hierarchical. That is, forecasts are required for different levels of aggregation: national, state, store, etc. 

## Experiment

The experiment consists of forecasting 28 days forward for 30,490 time series. The series correspond to the bottom level of the M5 competition. The evaluation is performed hierarchically. That is, the forecasts are aggregated to obtain the higher levels and subsequently evaluate the accuracy of each one of them.

The evaluation metric is the weighted Root Mean Scaled Squared Error (wRMSSE) defined as follows,

<img width="237" alt="image" src="https://user-images.githubusercontent.com/10517170/206323185-cad22076-cf91-40ea-a060-a83704bf790a.png">

Where the `RMSSE` is defined by,

<img width="648" alt="image" src="https://user-images.githubusercontent.com/10517170/206323251-7e2a3318-1f24-4f34-ad32-d872464454b4.png">

The `wRMSSE` is the official metric used in the M5 competition.


Detailed results per dataset are shown below.

## Results

### Performance

The table shows the performance of `StatsForecast` and `AmazonForecast` on the M5 dataset using the official evaluation metric.

<img width="637" alt="image" src="https://user-images.githubusercontent.com/10517170/206330119-48be0a7c-9ff6-412e-a52b-59a181c2a9d9.png">

The following table shows the performance across all levels of the hierarchy:

<img width="637" alt="image" src="https://user-images.githubusercontent.com/10517170/206330159-1497b625-fc70-4b91-af96-cf52240cc9e6.png">

### Time

Excluding time spent in the console and just accounting for processing and computing time, Amazon Forecast took 4.1 hours to run. In comparison, StatsForecast took just 15 minutes to run. Running time accounts for the end-to-end pipeline including loading data, training and forecasting. 

### Cost 

Amazon included a cost calculator that is quite accurate. The estimated cost was 803.53     USD. 

In comparison, we paid 1.2 USD of EC2 associated costs. (This could have been further reduced by using spot instances.)

Below, you can find the detailed results. 

**Speed and Cost on M5 Data set**

| Model | Time (hours) | Cost (USD) |
|:-------|-------------:|-----:|
| StatsForecast | 0.26 |  1.2    |
| AmazonForecast | 4.1 | 803.53 |

## Conclusions

In conclusion: for this setting, in terms of speed, costs, simplicity and accuracy, AutoML is far behind simple statistical methods. In terms of accuracy, they seem to be rather close.

This conclusion might or not hold in other datasets, however, given the a priori uncertainty of the benefits and the certainty of cost, statistical methods should be considered the first option in daily forecasting practice.

Although this experiment does not focus on comparing machine learning and deep learning vs statistical methods, it supports our [previous conclusions](https://github.com/Nixtla/statsforecast/tree/main/experiments/m3) on the current validity of simpler methods for many forecasting tasks.

## Unsolicited Advice
Choose your models wisely.

It would be extremely expensive and borderline irresponsible to favor AutoML in an organization before establishing solid baselines.

Simpler is sometimes better. Not everything that glows is gold.

Go and try other great open-source libraries like GluonTS, Darts and Sktime.

## Reproducibility
You can fully reproduce the experiment by following [this step-by-step notebook](https://nixtla.github.io/statsforecast/examples/aws/statsforecast.html).
