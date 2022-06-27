# Fugue experiments

[Fugue](https://github.com/fugue-project/fugue) is an abstraction layer letting you fully utilize
computing engines in simple and unified approaches. Currently Fugue support Spark and Dask, so
in this experiment, we test the performance of Spark and Dask on top of Databricks.

**Notice**: for this entire test, we only need the sequential part of `StatsForecast` (`n_jobs=1`). That means, if using Fugue, theoretically, the non-sequential part of the code can be removed.

## Main results

Fugue + |n time series |     time (mins) | cluster size (n cpus) |
|-----------:|-----------:|---------:|-------:|
| Spark |  10,000     |  2.16 |   128 |
| Dask |  10,000     |  3.2 |   128 |

## Reproducibility

To partially reproduce the main results you have:

1. Create a databricks cluster according to the [config json](databricks_conf.json)
2. Run `nixtla.ipynb`

Our init script to start Dask on Databricks is not included. Our Dask setup on Databricks
may not be optimal.