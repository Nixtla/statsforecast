# Statistical vs Deep Learning forecasting methods
Comparison of several Deep Learning models and ensembles to classical statistical univariate models for the 3,003 series of the M3 competition.

## Abstract

We present a reproducible experiment that shows that:

1. A simple statistical ensemble outperforms most individual deep-learning models. 

2. A simple statistical ensemble is 25,000 faster and only slightly less accurate than an ensemble of deep learning models.

In other words, deep-learning ensembles outperform statistical ensembles just by 0.36 points in SMAPE. However, the DL ensemble takes more than 14 days to run and costs around USD 11,000, while the statistical ensemble takes 6 minutes to run and costs $0.5c. 


## Background

In [Statistical, machine learning and deep learning forecasting methods: Comparisons and ways forward](https://www.tandfonline.com/doi/full/10.1080/01605682.2022.2118629), Makridakis and other prominent participants of the forecasting science community compare several Deep Learning and Statistical models for all 3,003 series of the M3 competition.

> The purpose of [the] paper is to test empirically the value currently added by Deep Learning (DL) approaches in time series forecasting by comparing the accuracy of some state-of-theart DL methods with that of popular Machine Learning (ML) and statistical ones.

The authors conclude that:

> We find that combinations of DL models perform better than most standard models, both statistical and ML, especially for the case of monthly series and long-term forecasts.

We don't think that's the full picture.

By including a statistical ensemble, we show that these claims are not completely warranted and that one should rather conclude that, for this setting at least, Deep Learning is rather unattractive.

## Experiment

Building upon the original design, we further included [A simple combination of univariate models](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300585) in the comparison. 

This ensemble is formed by averaging four statistical models: [`AutoARIMA`](https://www.jstatsoft.org/article/view/v027i03), [`ETS`](https://robjhyndman.com/expsmooth/), [`CES`](https://onlinelibrary.wiley.com/doi/full/10.1002/nav.22074) and [`DynamicOptimizedTheta`](https://doi.org/10.1016/j.ijforecast.2016.02.005). This combination won sixth place and was the simplest ensemble among the top 10 performers in the M4 competition. 
 
For the experiment, we use StatsForecast's implementation of [Arima](https://nixtla.github.io/statsforecast/models.html#autoarima), [ETS](https://nixtla.github.io/statsforecast/models.html#autoets), [CES](https://nixtla.github.io/statsforecast/models.html#autoces) and [DOT](https://nixtla.github.io/statsforecast/models.html#dynamic-optimized-theta-method). 

For the DL models and ensembles, we reproduce the reported metrics and results from the mentioned paper.

## Results

### Accuracy: Comparison with SOTA benchmarks 
Accuracy is reported in Symmetric mean absolute percentage error ([SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error))

The M3 dataset has four groups of time series. In the next graph, you can see the performance of all models and ensembles.

<img width="734" alt="image" src="https://user-images.githubusercontent.com/10517170/204959437-6a124ad6-a1b5-47c7-ba24-91d447efb1ce.png">

In the next table, you can see the performance of the models across all four groups and the average performance for all groups.

<img width="734" alt="image" src="https://user-images.githubusercontent.com/10517170/204958689-38cdea5f-58d7-42f5-b825-0f2a0b63f617.png">

### Computational Complexity: Comparison with SOTA benchmarks 

Computational complexity is reported in time, lines of code and, Relative Computational Complexity (RCC). 

#### Time
Using `StatsForecast` and a 96 cores EC2 instance (c5d.24xlarge) it takes 5.6 mins to train, forecast and ensemble the four models for the 3,003 series of M3. 

| Time (mins) | Yearly | Quarterly | Monthly | Other |
|-----|-------:|-------:|--------:|--------:|
|StatsForecast ensemble| 1.10 | 1.32 | 2.38 | 1.08 |

The authors of the paper only report computational time for the monthly group, which amounts to 20,680 mins or 14.3 days. In comparison, the StatsForecast ensemble only takes 2.38 minutes to run for that group. Furthermore, the authors don't include times for Hyperparameter optimization. 

For this comparison, we will take the reported 14 days of computational time. However, it must be noted that the true computational time must be significantly higher for all groups. 

#### Engineering 

Furthermore, running all statistical models, including data downloading, data wrangling, training, forecasting and ensembling the models, can be achieved in less than 150 lines of Python code. In comparison, [this](https://github.com/gjmulder/m3-gluonts-ensemble) repo has more than 1,000 lines of code and needs Python, R, Mongo and Shell code.

#### Relative Computational Complexity 
The mentioned paper uses Relative Computational Complexity (RCC) for comparing the models. To calculate the RCC of `StatsForecast`, we followed the same methodology and measured the time it takes to generate naive forecasts for all 3,003 series in our environment. 

Using a `c5d.24xlarge` instance (96 CPU, 192 GB RAM) it takes 12 seconds to train and predict 3,003 instances of a Seasonal Naive forecast. Therefore, the RCC of the simple ensemble is 28. 

In the next table, you can find the RCC of the deep learning models and the ensembles

| Method | Type | Relative Computational Complexity (RCC)|
|--------|------|----------------------------------------:|
|DeepAR| DL |313,000|
|Feed-Forward| DL  |47,300 |
|Transformer| DL | 47,500 |
|WaveNet| DL | 306,000 |
|Ensemble-DL | DL | 713,800 |
|Ensemble - Stats | Statistical | 28 |
|SeasonalNaive| Benchmark | 1 | 


### Summary: Comparison with SOTA benchmarks

We present a summary comparison, including SMAPE, RCC, Cost proxy, and self-reported computational time. 

<img width="734" alt="image" src="https://user-images.githubusercontent.com/10517170/204958747-ea9e53ce-d0fc-41d1-bb71-eac7bed4be94.png">

We observe that `StatsForecast` yields average SMAPE results similar to DeepAR with computational savings of 99%.

Furthermore, the StatsForecast ensemble:
- Has better performance than the `N-BEATS` model for the `Yearly` and `Other` groups.
- Has a better average performance than the individual `Gluon-TS` models. 
- It performs better than all `Gluont-TS` models for the `Monthly` and `Other` groups. 
- It is consistently better than the `Transformer`, `Wavenet`, and `Feed-Forward` models.

In conclusion, the deep learning ensemble achieves 12.27 points of accuracy (sMAPE), with a relative computational cost of 713,000 and a proxy monetary cost of USD 11,4200. 
The simple statistical ensemble achieves 12.63 points of accuracy, with a relative computational cost of 28 and a proxy monetary cost of USD 0.5c.

Therefore, the DL Ensemble is only 0.36 points more accurate than the statistical ensemble, but 25,000 times more expensive. 

In plain English: a deep-learning ensemble that takes more than 14 days to run and costs around USD 11,000, outperforms a statistical ensemble that takes 6 minutes to run and costs $0.5c by only 0.36 points of SMAPE. 


## Conclusions
For this setting: Deep Learning models are simply worse than a statistical ensemble. To outperform this statistical ensemble by 0.36 points of SMAPE a complicated deep learning ensemble is needed. The deep learning ensemble, however, takes more than two weeks to run, costs several thousands of dollars and demands several engineering hours. 

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

- [Hyndman, Rob J. & Khandakar, Yeasmin (2008). "Automatic Time Series Forecasting: The forecast package for R"](https://www.jstatsoft.org/article/view/v027i03)
- [Hyndman, Rob J., et al (2008). "Forecasting with exponential smoothing: the state space approach"](https://robjhyndman.com/expsmooth/)
- [Svetunkov, Ivan & Kourentzes, Nikolaos. (2015). "Complex Exponential Smoothing". 10.13140/RG.2.1.3757.2562. ](https://onlinelibrary.wiley.com/doi/full/10.1002/nav.22074)
- [Jose A. Fiorucci, Tiago R. Pellegrini, Francisco Louzada, Fotios Petropoulos, Anne B. Koehler: Models for optimising the theta method and their relationship to state space models, International Journal of Forecasting, Volume 32, Issue 4, 2016, Pages 1151-1161, ISSN 0169-2070](https://doi.org/10.1016/j.ijforecast.2016.02.005)
- [Fotios Petropoulos, Ivan Svetunkov: A simple combination of univariate models, International Journal of Forecasting, Volume 36, Issue 1, 2020, Pages 110-115, ISSN 0169-2070.](https://doi.org/10.1016/j.ijforecast.2019.01.006)
- [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, ArtemiosAnargyros Semenoglou, Gary Mulder & Konstantinos Nikolopoulos (2022): Statistical, machine
learning and deep learning forecasting methods: Comparisons and ways forward, Journal of the
Operational Research Society, DOI: 10.1080/01605682.2022.2118629](https://www.tandfonline.com/doi/pdf/10.1080/01605682.2022.2118629?needAccess=true)


