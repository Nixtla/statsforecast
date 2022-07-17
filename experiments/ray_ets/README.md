# ray experiments

## Main results


### Non-seasonal time series

|n time series |     time (mins) | cluster size (n cpus) |
|-----------:|---------:|-------:|
|  10,000     |  2.20 |   2,000 |
| 100,000     |  2.49 |   2,000 |
| 500,000     | 3.34  |   2,000 |
| 1,000,000 | 4.62  |   2,000 |
| 2,000,000 | 6.01  |   2,000 |

### Seasonal time series

|n time series |     time (mins) | cluster size (n cpus) |
|-----------:|---------:|-------:|
|  10,000     |  2.4 |   2,000 |
| 100,000     |  4.0 |   2,000 |
| 500,000     | 12.60  |   2,000 |
| 1,000,000 | 25.56  |   2,000 |
| 2,000,000 | 48.73  |   2,000 |

## Reproducibility

To reproduce the main results you have:

1. Create a conda environment using `conda env create -f environment.yml`.
2. Activate the environment using `conda actiate ets_ray`.
3. Run `ray up cluster.yaml` to launch a `ray` cluster and get the `ray address`. 
4. Run the experiments using `python -m experiment --ray-address [your_ray_address] --seasonality [0, 7]` (`0` for non-seasonal models, `7` for seasonal models).
5. Finally shut down the cluster using `ray down cluster.yaml`.
