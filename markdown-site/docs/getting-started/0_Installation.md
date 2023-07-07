---
title: Install
---

> Install StatsForecast with pip or conda

You can install the *released version* of `StatsForecast` from the
[Python package index](https://pypi.org) with:

``` python
pip install statsforecast
```

or

``` python
conda install -c conda-forge statsforecast
```

:::warning

We are constantly updating StatsForecast, so we suggest fixing the
version to avoid issues. `pip install statsforecast=="1.0.0"`

:::


:::tip

We recommend installing your libraries inside a python virtual or [conda
environment](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html).

:::

#### User our env (optional) {#user-our-env-optional}

If you don’t have a Conda environment and need tools like Numba, Pandas,
NumPy, Jupyter, StatsModels, and Nbdev you can use ours by following
these steps:

1.  Clone the StatsForecast repo:

``` bash
$ git clone https://github.com/Nixtla/statsforecast.git && cd statsforecast
```

1.  Create the environment using the `environment.yml` file:

``` bash
$ conda env create -f environment.yml
```

1.  Activate the environment:

``` bash
$ conda activate statsforecast
```

