# MFLES
A method to forecast time series based on Gradient Boosted Time Series Decomposition which treats traditional decomposition as the base estimator in the boosting process. Unlike normal gradient boosting, slight learning rates are applied at the component level (trend/seasonality/exogenous).
        
The method derives its name from some of the underlying estimators that can enter into the boosting procedure, specifically: a simple Median, Fourier functions for seasonality, a simple/piecewise Linear trend, and Exponential Smoothing.

## Gradient Boosted Time Series Decomposition Theory
The idea is pretty simple, take a process like decomposition and view it as
a type of 'psuedo' gradient boosting since we are passing residuals around
simlar to standard gradient boosting. Then apply gradient boosting approaches
such as iterating with a global mechanism to control the process and introduce
learning rates for each of the components in the process such as trend or
seasonality or exogenous. By doing this we graduate from this 'psuedo' approach
to full blown gradient boosting.

## Some Benchmarks
Average SMAPE from a few M4 datasets
| Dataset    | AutoMFLES | AutoETS |
| -------- | ------- | ------- |
| Monthly  | 12.84    | 13.59 |
| Hourly | 11.81     | 17.19 |
| Weekly    | 8.32    | 8.64 |
| Quarterly    | 10.66    | 10.26 |
