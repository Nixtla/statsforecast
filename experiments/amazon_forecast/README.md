---
order: 1
title: Amazon Forecast vs StatsForecast
listing:
  fields: [title, description]
  type: table
  sort-ui: true
  filter-ui: true 
---

## Amazon Forecast vs open source statistical methods
>Tl:Dr: We paid USD $1,404.00 and spend 4 hours in the AWS console so you don't have to. 

In this reproducible experiment, we compare Amazon Forecast and StatsForecast. Given the prominent use of AWS Forecast in demand forecasting, for this experiment, we used the 30,490 series of daily sales at Walmart from the M5 competition. 


### Amazon Forecast
Amazon Forecast is a time-series forecasting service based on machine learning (ML) and built for business metrics analysis. 

```
It uses machine learning (ML) to generate more accurate demand forecasts with just a few clicks, without requiring any prior ML experience. Amazon Forecast includes algorithms that are based on over twenty years of forecasting experience and developed expertise used by Amazon.com bringing the same technology used at Amazon to developers as a fully managed service, removing the need to manage resources. Amazon Forecast uses ML to learn not only the best algorithm for each item, but the best ensemble of algorithms for each item, automatically creating the best model for your data.
```

Amazon Forecast is one of the leading forecasting services out there. You can read more about its features and pricing tiers here. 

### StatsForecast
StatsForecast is an open-source python library from Nixtla. Nixtla offers a wide variety of statistical methods.
LLENAR

### Conclusions

Amazon Forecast achieved XXX in accruacy, took XXX hours to run and cost XXX usd. Statsforecast, on a XXX machine, achieved XXX in accuracy, took XXX minutes to run and cost only XXX cents. 

For this data set, we show therefore that: 

* Amazon Forecast is 30% less accurate and x times more expensive than running an open-source alternative in a cloud server. 
* Machine Learning methods are outperformed by classical methods in terms of speed, accuracy and cost. 

Although using StatsForecast requires some basic knowledge of Python and cloud computing, the results are simply better. 


### Data
M5 description. XXX

## Experiment

XXX Experimetnal setting:
* Horzions
* Error metrics, bla bla

We ran the Amazon forecast according to the the instructions and forecasted 3 quantiles for the next x days.
We spined of a xxx EC2 instance and run statsforecast with just X models. 

Detailed results per dataset are shown below.

## Results

## Performance

SUMMARY OF PERFOMANCE

The table shows the performance of `StatsForecast` and `AmazonForecast` on the M5 dataset using the official evaluation metric.

<img width="623" alt="image" src="https://user-images.githubusercontent.com/10517170/206248653-3ed6e9eb-9bce-4f6f-8d4b-7ade3c305ce5.png">

The following table shows the performance across all levels of the hierarchy:

<img width="623" alt="image" src="https://user-images.githubusercontent.com/10517170/206248740-d84455b4-451e-4bc8-b9ae-6b7d9010bb1d.png">


### Time

Excluding time spent in the console and just accounting for processing and computing time, Amazon Forecast took 6.3 hours to run. In comparison, StatsForecast took just 48 minutes to run. XXX

Running time includes the end-to-end pipeline including loading data, training and forecasting. 

### Cost 

Amazon included a cost calculator that is quite accurate. We fill in the followig data corresponding to the experiment. XXX
The estimated cost was XXX dolars. 

In comparison, we paid XXX of EC2 associated costs. (This could have been further reduced by using spot instances)

Below, you can find the detailed results. 

**Speed and Cost on M5 Data set**

| Model | Time (hours) | Cost (USD) |
|:-------|-------------:|-----:|
| StatsForecast | 0.26 |  1.2    |
| AmazonForecast | 4.1 | 795.54 |


## Conclusions

Although this experiment does not focus on comparing machine learning and deep learning vs statistical methods, it supports our previous conclusions on the current validity of simpler methods for many forecasting tasks.

XXX


## Unsolicited Advice
Don't always grab for the low-hanging fruit. It pays off. 

## Reproducibility
You can fully reproduce the experiment by following this step-by-step notebook.