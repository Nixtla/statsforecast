# MSTL is faster and more accurate than Prophet and NeuralProphet with multiple seasonalities

### Install StatsForecast
```bash
pip install statsforecast
```

## Reproducibility


1. Create a conda environment `exp_mstl` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate exp_mstl
  ```

4. Run the experiments for each dataset and each model using 
  ```shell
  python -m src.main
  ```

## Misc.

* [`StatsForecast`](https://github.com/nixtla/statsforecast) also includes a variety of lightning fast baseline models.
* If you really need to do forecast at scale, [here](https://github.com/nixtla/statsforecast/tree/main/experiments/ray) we show how to forecast 1 million time series under 30 minutes using [Ray](https://github.com/ray-project/ray).
* If you are interested in SOTA Deep Learning models, check [`NeuralForecast`](https://github.com/nixtla/neuralforecast)


