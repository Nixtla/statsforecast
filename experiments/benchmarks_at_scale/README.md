# Benchmarks at Scale

## Main results

### Computational time

| N time series   |   Time (mins) |   N cpus |   CVWindows | Cost (Dollars)   |
|:----------------|--------------:|---------:|------------:|:-----------------|
| 10,000          |          0.32 |      128 |           7 | $0.14            |
| 100,000         |          0.74 |      128 |           7 | $0.33            |
| 1,000,000       |          4.81 |      128 |           7 | $2.14            |
| 5,000,000       |         21.87 |      128 |           7 | $9.73            |
| 10,000,000      |         44.12 |      128 |           7 | $19.63           |

### Performace (MSE)

| N time series   |   Croston |   SeasNaive |   Naive |   ADIDA |   HistoricAverage |   SeasWindowAverage |   iMAPA |   WindowAverage |   SeasExpSmooth |
|:----------------|----------:|------------:|--------:|--------:|------------------:|--------------------:|--------:|----------------:|----------------:|
| 10,000          |    4.1045 |      0.0414 |  8.0418 |  4.1366 |            4.0313 |              0.026  |  4.1366 |          4.0239 |          8.0377 |
| 100,000         |    4.1035 |      0.0418 |  8.0403 |  4.1373 |            4.0307 |              0.0261 |  4.1373 |          4.0233 |          8.0372 |
| 1,000,000       |    4.1046 |      0.0417 |  8.0417 |  4.1381 |            4.0314 |              0.026  |  4.1381 |          4.024  |          8.038  |
| 5,000,000       |    4.1042 |      0.0417 |  8.0416 |  4.1377 |            4.0311 |              0.026  |  4.1377 |          4.0237 |          8.038  |
| 10,000,000      |    4.1043 |      0.0417 |  8.0418 |  4.1379 |            4.0313 |              0.026  |  4.1379 |          4.0239 |          8.0381 |

## Reproducibility

To reproduce the main results you have:
1. Install the conda environment using,

```bash
conda env create -f environment.yml
```

2. Activate the environment using,

```bash
conda activate benchmarks_at_scale
```

3. Generate the data using,

```bash
python -m src.data
```

4. Run the experiments using,

```bash
python -m src.experiment
```
