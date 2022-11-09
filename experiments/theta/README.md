# Theta Model

The theta family of models has been shown to perform well in various datasets such as M3. `StatsForecast` includes four models of the family: `Theta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta`. The implementation is based on the [work of Jose A. Fioruccia, Tiago R. Pellegrini, Francisco Louzada, Fotios Petropoulos, and Anne B.Koehlerf](https://www.sciencedirect.com/science/article/pii/S0169207016300243), and optimized using numba.


## Main Results

### Comparison with previous implementations

The original implementation was originally developed for R through the [`forecTheta`](https://cran.r-project.org/web/packages/forecTheta/index.html) package. The following table shows a comparison for the M3 dataset of the StatForecast implementation with the one reported in the [original paper](https://www.sciencedirect.com/science/article/pii/S0169207016300243). As can be seen, `StatsForecast` gives very similar results to the original implementation.

<img width="735" alt="image" src="https://user-images.githubusercontent.com/10517170/200728321-e40a3db5-2cf1-486e-af75-9d841902fc44.png">

In terms of computational cost, the `StatsForecast` implementation is very efficient. The following table shows that the whole experiment took only 4 minutes.

| Time (mins) | Yearly | Quarterly | Monthly | Other |
|-----|-------:|-------:|--------:|--------:|
|Theta| 0.23 | 0.21 | 0.23 | 0.22 |
|OptimizedTheta | 0.23 | 0.21 | 0.24 | 0.23 |
|DynamicTheta | 0.23 | 0.207 | 0.23 | 0.22 |
|DynamicOptimizedTheta | 0.23 | 0.21 | 0.24 | 0.23 | 


### Comparison with SOTA benchmarks

In the paper [Statistical, machine learning and deep learning forecasting methods: Comparisons and ways forward](https://www.tandfonline.com/doi/full/10.1080/01605682.2022.2118629), the use of Deep Learning models in the M3 dataset is studied. The paper uses as a statistical benchmark the ensemble between the `AutoARIMA` and `ETS` models. Using `StatsForecast`, we included the `DynamicOptimizedTheta` model in addition to `AutoARIMA` and `ETS` in the ensemble. The results are shown below (`N-BEATS` and `Gluon-TS` results were taken from the original paper). 

<img width="748" alt="image" src="https://user-images.githubusercontent.com/10517170/200728018-7bb655cf-c2ba-4724-945e-13dbe4db1730.png">

We can see that the StatsForecast ensemble:
- Has better performance than the `N-BEATS` model for the yearly frequency.
- Has a better average performance than the individual `Gluon-TS` models.
- It is consistently better than the `Transformer`, `Wavenet`, and `Feed-Forward` models.
- It performs better than all `Gluont-TS` models for the monthly frequency. 

In terms of computational cost, `StatsForecast` generated the ensemble in only 6 minutes as shown in the table below,

| Time (mins) | Yearly | Quarterly | Monthly | Other |
|-----|-------:|-------:|--------:|--------:|
|StatsForecast ensemble| 1.03 | 1.32 | 2.38 | 1.01 |

The mentioned paper also shows the computational cost of the models used. To ensure the comparability of the results, the relative computational complexity (RCC) is shown below. To calculate the RCC of `StatsForecast`, we took the time to generate naive forecasts in the same environment.

| Method | Type | Relative Computational Complexity (RCC)|
|--------|------|----------------------------------------:|
|DeepAR| DL |313,000|
|Feed-Forward| DL  |47,300 |
|Transformer| DL | 47,500 |
|WaveNet| DL | 306,000 |
|Ensemble-DL | DL | 713,800 |
|StatsForecast | Statistical | 27 |

We observe that `StatsForecast` yields average SMAPE results similar to DeepAR with computational savings of 99%.


## References

- [Jose A. Fiorucci, Tiago R. Pellegrini, Francisco Louzada, Fotios Petropoulos, Anne B. Koehler: Models for optimising the theta method and their relationship to state space models, International Journal of Forecasting, Volume 32, Issue 4, 2016, Pages 1151-1161, ISSN 0169-2070](https://doi.org/10.1016/j.ijforecast.2016.02.005.)
- [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, ArtemiosAnargyros Semenoglou, Gary Mulder & Konstantinos Nikolopoulos (2022): Statistical, machine
learning and deep learning forecasting methods: Comparisons and ways forward, Journal of the
Operational Research Society, DOI: 10.1080/01605682.2022.2118629](https://www.tandfonline.com/doi/pdf/10.1080/01605682.2022.2118629?needAccess=true)


## Reproducibility

To reproduce the main results you have to:

1. Create the environment using `conda env create -f environment.yml`. 
2. Activate the environment using `conda activate theta`.
3. Run the experiments using `python -m src.theta --dataset M3 --group [group] --model [model]` where `[model]` can be `Theta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta`, `ThetaEnsemble`, and `[group]` can be `Other`, `Monthly`, `Quarterly`, and `Yearly`.
4. Finally you can evaluate the forecasts using `python -m src.evaluation`.
