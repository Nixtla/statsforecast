# Comparison of Statistical and deep learning forecasting methods in M3. 


## Abstract

We present a reproducible experiment that shows that:

1. A simple ensemble of univariate models outperforms individual deep learning models and achieves so-called SOTA.

2. A simple ensemble of univariate models is 25,000 more efficient and only slightly less accurate than an ensemble of deep learning models.


## Background

In [Statistical, machine learning and deep learning forecasting methods: Comparisons and ways forward](https://www.tandfonline.com/doi/full/10.1080/01605682.2022.2118629), prominent participants of the forecasting science community compare Deep Learning models and Statistical models for all 3,003 series of the M3 competition.

```
The purpose of this paper is to test empirically the value currently added by Deep Learning
(DL) approaches in time series forecasting by comparing the accuracy of some state-of-theart DL methods with that of popular Machine Learning (ML) and statistical ones.
```
In particular, the authors use Gluon TS and Nbeats. The study was funded through an AWS Machine Learning Research Award with funds, AWS credits and hands-on sessions with Amazon scientists and engineers. 

The authors conclude that:

```
We find that combinations of DL models perform better than most standard models, both statistical and
ML, especially for the case of monthly series and long-term forecasts.
```

We don't agree.

By simply including a combination of four classical univariate models we show that these claims are in no way warranted and that one should rather conclude that, for this setting at least, Deep Learning is second best to statistical univariate models.

## Experiment

Following the original design, we included [A simple combination of univariate models](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300585). 

This ensemble is formed by averaging four models: `ARIMA`, `ETS`, `CES` and `DynamicOptimizedTheta`.
This combination won sixth place and was the simplest ensamble among the top 10 performers in the M4 competition. 
 
For the experiment, we use StatsForecast's implementation of Arima, ETS, CES and DOT. 

For the DL models, we reproduce the reported metrics and results from the mentioned paper and included

## Results

### Accuracy: Comparison with SOTA benchmarks 


<img width="717" alt="image" src="https://user-images.githubusercontent.com/10517170/200745682-0cf03ab0-5b54-409a-a5fd-75a3925a33ec.png">

The results are shown below (`N-BEATS` and `Gluon-TS` results were taken from the original paper).


### Computational Complexity: Comparison with SOTA benchmarks 

Using `StatsForecast` and a 96 cores EC2 instance (XXXX) it takes 6 mins to train, forecast and ensemble the four models for the 3,003 series of M3. 

| Time (mins) | Yearly | Quarterly | Monthly | Other |
|-----|-------:|-------:|--------:|--------:|
|StatsForecast ensemble| 1.10 | 1.32 | 2.38 | 1.08 |


Furthermore, this experiment including downloading, data wrangling, training, forecasting and ensembling the models, can be achieved in less than 150 lines of Python code. (In comparison, [This](https://github.com/gjmulder/m3-gluonts-ensemble) repo](https://github.com/gjmulder/m3-gluonts-ensemble) has more than 1,000 lines of code and needs Python, R, Mongo and Shell code)

The mentioned paper uses Relative Computational Complexity (RCC) for comparing the models. To calculate the RCC of `StatsForecast`, we took the time to generate naive forecasts in the same environment. 

Using XXX it takes 12 seconds to train and predict 3,003 instances of a Naive forecast. Therefore, the RCC of the simple ensemble is 28.  [REVISAR FEDE!!!]

| Method | Type | Relative Computational Complexity (RCC)|
|--------|------|----------------------------------------:|
|DeepAR| DL |313,000|
|Feed-Forward| DL  |47,300 |
|Transformer| DL | 47,500 |
|WaveNet| DL | 306,000 |
|Ensemble-DL | DL | 713,800 |
|StatsForecast | Statistical | 28 |
|Naive| Naive| 1 | 



### Costs: Comparison with SOTA benchmarks 
In actual use cases, the cost of computation also should be considered. Given the hourly cost of the setting we used, the table can be "translated" to dollars. 


### Summary: Comparison with SOTA benchmarks

We observe that `StatsForecast` yields average SMAPE results similar to DeepAR with computational savings of 99%.

* On average: 
 * the deep learning ensemble achieves XXX of accuracy, with a Computational cost of 713,000 and a proxy monetary cost of X.
 * The simple univariate ensemble achieves XXX of accuracy, with a computational cost of 28 and a proxy monetary cost of X. 

 That means, DL is only 0.16 points more accurate than Statistical Models, but 25,000 times more expensive. 


We can see that the StatsForecast ensemble:
- Has better performance than the `N-BEATS` model for the yearly and other groups.
- Has a better average performance than the individual `Gluon-TS` models. In particular, the ensemble is better than XXX for all 4 groups. XXX
- It is consistently better than the `Transformer`, `Wavenet`, and `Feed-Forward` models.
- It performs better than all `Gluont-TS` models for the monthly and other groups. 



## Conclusions
For this setting: Deep Learning models are simply worse or marginally better than simpler univariate models in terms of accuracy. In terms of speed, costs, simplicity and interpretability, deep learning still has a huge debt to the field. 

This conclusion might or not hold in other datasets, however, given the a priori uncertainty of the benefits and the certainty of cost, statistical methods should be yet considered the first option in daily forecasting practice. 

## Unsolicited Advice

Choose your models wisely. 

It would be extremely expensive and borderline irresponsible to favor deep learning models in an organization before establishing solid baselines. 

Simpler is sometimes better. Not everything that glows is gold. 




## References

- [Jose A. Fiorucci, Tiago R. Pellegrini, Francisco Louzada, Fotios Petropoulos, Anne B. Koehler: Models for optimising the theta method and their relationship to state space models, International Journal of Forecasting, Volume 32, Issue 4, 2016, Pages 1151-1161, ISSN 0169-2070](https://doi.org/10.1016/j.ijforecast.2016.02.005)
- [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, ArtemiosAnargyros Semenoglou, Gary Mulder & Konstantinos Nikolopoulos (2022): Statistical, machine
learning and deep learning forecasting methods: Comparisons and ways forward, Journal of the
Operational Research Society, DOI: 10.1080/01605682.2022.2118629](https://www.tandfonline.com/doi/pdf/10.1080/01605682.2022.2118629?needAccess=true)
- [Fotios Petropoulos, Ivan Svetunkov: A simple combination of univariate models, International Journal of Forecasting, Volume 36, Issue 1, 2020, Pages 110-115, ISSN 0169-2070.](https://doi.org/10.1016/j.ijforecast.2019.01.006)


## Reproducibility

To reproduce the main results you have to:

1. Create the environment using `conda env create -f environment.yml`. 
2. Activate the environment using `conda activate theta`.
3. Run the experiments using `python -m src.theta --dataset M3 --group [group] --model [model]` where `[model]` can be `Theta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta`, `ThetaEnsemble`, and `[group]` can be `Other`, `Monthly`, `Quarterly`, and `Yearly`.
4. Finally you can evaluate the forecasts using `python -m src.evaluation`.
