# Statistical vs Deep Learning forecasting methods for time series

## Abstract

We present a reproducible experiment that shows that:

1. A simple statistical ensemble outperforms most individual deep-learning models. 

2. A simple statistical ensemble is 25,000 faster and only slightly less accurate than an ensemble of deep learning models.

In other words, deep-learning ensembles outperform statistical ensembles by 0.36 points in SMAPE; but the DL ensemble (12.27 SMAPE) takes more than 15 days to run and costs around USD 11,000, while the statistical ensemble (12.63) takes 6 minutes to run and costs $0.5c. 


## Background

In [Statistical, machine learning and deep learning forecasting methods: Comparisons and ways forward](https://www.tandfonline.com/doi/full/10.1080/01605682.2022.2118629), Makridakis and other prominent participants of the forecasting science community compare Deep Learning models and Statistical models for all 3,003 series of the M3 competition.

> The purpose of [the] paper is to test empirically the value currently added by Deep Learning (DL) approaches in time series forecasting by comparing the accuracy of some state-of-theart DL methods with that of popular Machine Learning (ML) and statistical ones.

The authors conclude that:

> We find that combinations of DL models perform better than most standard models, both statistical and ML, especially for the case of monthly series and long-term forecasts.


We don't think that's the full picture.

By including a statistical ensemble, we show that these claims are not completely warranted and that one should rather conclude that, for this setting at least, Deep Learning is rather unattractive.

## Experiment

Building upon the original design, we further included [A simple combination of univariate models](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300585) in the comparison. 

This ensemble is formed by averaging four statistical models: `ARIMA`, `ETS`, `CES` and `DynamicOptimizedTheta`. This combination won sixth place and was the simplest ensemble among the top 10 performers in the M4 competition. 
 
For the experiment, we use StatsForecast's implementation of Arima, ETS, CES and DOT. 

For the DL models, we reproduce the reported metrics and results from the mentioned paper.

## Results

### Accuracy: Comparison with SOTA benchmarks 

<img width="734" alt="image" src="https://user-images.githubusercontent.com/10517170/204958433-c216c651-3a12-46ec-88bf-fb1e580742cc.png">
<img width="734" alt="image" src="https://user-images.githubusercontent.com/10517170/204958689-38cdea5f-58d7-42f5-b825-0f2a0b63f617.png">

### Computational Complexity: Comparison with SOTA benchmarks 

Using `StatsForecast` and a 96 cores EC2 instance (c5d.24xlarge) it takes 5.6 mins to train, forecast and ensemble the four models for the 3,003 series of M3. 

| Time (mins) | Yearly | Quarterly | Monthly | Other |
|-----|-------:|-------:|--------:|--------:|
|StatsForecast ensemble| 1.10 | 1.32 | 2.38 | 1.08 |


Furthermore, this experiment including downloading, data wrangling, training, forecasting and ensembling the models, can be achieved in less than 150 lines of Python code. In comparison, [this](https://github.com/gjmulder/m3-gluonts-ensemble) repo has more than 1,000 lines of code and needs Python, R, Mongo and Shell code.

The mentioned paper uses Relative Computational Complexity (RCC) for comparing the models. We stick to that metric for comparability's sake. To calculate the RCC of `StatsForecast`, we measured the time it takes to generate naive forecasts for all 3,0003 series in the same environment. 

Using a `c5d.24xlarge` instance (96 CPU, 192 GB RAM) it takes 12 seconds to train and predict 3,003 instances of a Seasonal Naive forecast. Therefore, the RCC of the simple ensemble is 28. 

| Method | Type | Relative Computational Complexity (RCC)|
|--------|------|----------------------------------------:|
|DeepAR| DL |313,000|
|Feed-Forward| DL  |47,300 |
|Transformer| DL | 47,500 |
|WaveNet| DL | 306,000 |
|Ensemble-DL | DL | 713,800 |
|StatsForecast | Statistical | 28 |
|SeasonalNaive| Statistical | 1 | 


### Summary: Comparison with SOTA benchmarks

In real-world use cases, the cost of computation also plays a role and should be considered. In the next table, you can see the summarized results for all models and ensembles. We compare accuracy measured in SMAPE, RCC, Cost proxy, and self-reported computational time.  

<img width="734" alt="image" src="https://user-images.githubusercontent.com/10517170/204958747-ea9e53ce-d0fc-41d1-bb71-eac7bed4be94.png">

We observe that `StatsForecast` yields average SMAPE results similar to DeepAR with computational savings of 99%.


Furthermore, we can see that the StatsForecast ensemble:
- Has better performance than the `N-BEATS` model for the yearly and other groups.
- Has a better average performance than the individual `Gluon-TS` models. In particular, the ensemble is better than Feed-Forward, Transformer and Wavenet for all 4 groups.
- It is consistently better than the `Transformer`, `Wavenet`, and `Feed-Forward` models.
- It performs better than all `Gluont-TS` models for the monthly and other groups. 

The deep learning ensemble achieves 12.27 of accuracy, with a relative computational cost of 713,000 and a proxy monetary cost of 11,420 USD.
The simple statistical ensemble achieves 12.63 of accuracy, with a relative computational cost of 28 and a proxy monetary cost of 0.5 USD. 
Therefore, the DL Ensemble is only 0.36 points more accurate than the statistical ensemble, but 25,000 times more expensive. 

In plain English: a deep-learning ensemble that takes more than 15 days to run and costs around USD 11,000, outperforms a statistical ensemble that takes 6 minutes to run and costs $0.5c by only 0.36 points of SMAPE. 


## Conclusions
For this setting: Deep Learning models are simply worse than a statistical ensemble. To outperform this statistical ensemble by 0.36 points of SMAPE complicated deep learning is needed. The deep learning ensemble takes more than two weeks to run, several thousands of dollars and many engineering hours. 

In conclusion: in terms of speed, costs, simplicity and interpretability, deep learning is far behind the simple statistical ensemble. In terms of accuracy, they seem to be rather close.

This conclusion might or not hold in other datasets, however, given the a priori uncertainty of the benefits and the certainty of cost, statistical methods should be considered the first option in daily forecasting practice.

## Unsolicited Advice

Choose your models wisely. 

It would be extremely expensive and borderline irresponsible to favor deep learning models in an organization before establishing solid baselines. 

Simpler is sometimes better. Not everything that glows is gold. 


## Reproducibility

To reproduce the main results you have to:

1. Create the environment using `conda env create -f environment.yml`. 
2. Activate the environment using `conda activate m3-dl`.
3. Run the experiments using `python -m src.experiment --group [group]` where `[group]` can be `Other`, `Monthly`, `Quarterly`, and `Yearly`.
4. Finally, you can evaluate the forecasts using `python -m src.evaluation`.


## References

- [Jose A. Fiorucci, Tiago R. Pellegrini, Francisco Louzada, Fotios Petropoulos, Anne B. Koehler: Models for optimising the theta method and their relationship to state space models, International Journal of Forecasting, Volume 32, Issue 4, 2016, Pages 1151-1161, ISSN 0169-2070](https://doi.org/10.1016/j.ijforecast.2016.02.005)
- [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, ArtemiosAnargyros Semenoglou, Gary Mulder & Konstantinos Nikolopoulos (2022): Statistical, machine
learning and deep learning forecasting methods: Comparisons and ways forward, Journal of the
Operational Research Society, DOI: 10.1080/01605682.2022.2118629](https://www.tandfonline.com/doi/pdf/10.1080/01605682.2022.2118629?needAccess=true)
- [Fotios Petropoulos, Ivan Svetunkov: A simple combination of univariate models, International Journal of Forecasting, Volume 36, Issue 1, 2020, Pages 110-115, ISSN 0169-2070.](https://doi.org/10.1016/j.ijforecast.2019.01.006)

