# Theta Model


### Main Results


## References


## Reproducibility

To reproduce the main results you have:

1. Create the environment using `conda env create -f environment.yml`. 
2. Activate the environment using `conda activate theta`.
3. Run the experiments using `python -m src.theta --dataset M3 --group [group] --model [model]` where `[model]` can be `Theta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta`, `ThetaEnsemble`, and `[group]` can be `Other`, `Monthly`, `Quarterly`, and `Yearly`.
4. Finally you can evaluate the forecasts using `python -m src.evaluation`.
