# ray experiments

## Main results

|n time series |     time (mins) | cluster size (n cpus) |
|-----------:|---------:|-------:|
|  10,000     |  1.8 |   2,000 |
| 100,000     |  4.6 |   2,000 |
| 500,000     | 17.9  |   2,000 |
| 1,000,000 | 35.8  |   2,000 |
| 2,000,000 | 82.3  |   2,000 |

## Reproducibility

To reproduce the main results you have:

1. Install `StatsForecast` including `ray` using `pip install "statsforecast[ray]"`.
2. Run `ray up cluster.yaml` to launch a `ray` cluster and get the `ray address`. 
3. Run the experiments using `python -m experiment --ray-address [your_ray_address]`.
4. Finally shut down the cluster using `ray down cluster.yaml`.
