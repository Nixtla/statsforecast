---
title: Utils
---

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
%load_ext autoreload
%autoreload 2
```

</details>

:::

> The `core.StatsForecast` class allows you to efficiently fit multiple
> `StatsForecast` models for large sets of time series. It operates with
> pandas DataFrame `df` that identifies individual series and datestamps
> with the `unique_id` and `ds` columns, and the `y` column denotes the
> target time series variable. To assist development, we declare useful
> datasets that we use throughout all `StatsForecast`’s unit
> tests.<br><br>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import random
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from numba import njit
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt

from nbdev.showdoc import add_docs, show_doc
```

</details>

:::

# <span style="color:DarkBlue">1. Synthetic Panel Data </span> {#synthetic-panel-data}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def generate_series(n_series: int,
                    freq: str = 'D',
                    min_length: int = 50,
                    max_length: int = 500,
                    n_static_features: int = 0,
                    equal_ends: bool = False,
                    engine:str = 'pandas', 
                    seed: int = 0) -> Union[pd.DataFrame, pl.DataFrame]:
    """Generate Synthetic Panel Series.

    Generates `n_series` of frequency `freq` of different lengths in the interval [`min_length`, `max_length`].
    If `n_static_features > 0`, then each series gets static features with random values.
    If `equal_ends == True` then all series end at the same date.

    **Parameters:**<br>
    `n_series`: int, number of series for synthetic panel.<br>
    `min_length`: int, minimal length of synthetic panel's series.<br>
    `max_length`: int, minimal length of synthetic panel's series.<br>
    `n_static_features`: int, default=0, number of static exogenous variables for synthetic panel's series.<br>
    `equal_ends`: bool, if True, series finish in the same date stamp `ds`.<br>
    `freq`: str, frequency of the data, [panda's available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).<br>
    `engine`: str, engine to be used in DataFrame construction; NOTE: index does not exist in polars DataFrame

    **Returns:**<br>
    `freq`: pandas.DataFrame | polars.DataFrame, synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous.
    """

    available_engines = ['pandas', 'polars']
    if engine.lower() not in available_engines:
        raise ValueError(
                """{} is not a correct engine; available options: {}"""
                .format(engine, ", ".join(available_engines))
            )

    seasonalities = {'D': 7, 'M': 12}
    season = seasonalities[freq]
    
    rng = np.random.RandomState(seed)
    series_lengths = rng.randint(min_length, max_length + 1, n_series)
    total_length = series_lengths.sum()

    vals_dict:dict = {}

    # Unique id generator
    vals_dict['unique_id'] = np.concatenate([
        np.repeat(i, serie_length) for i, serie_length in enumerate(series_lengths)
    ])

    vals_dict['y'] = np.arange(total_length) % season + rng.rand(total_length) * 0.5

    # Generating X number of dates that will be concatenated over to create
    # continues repetition for each unique_id, 'ds' column will be the final
    # result.
    dates = (
        pd.date_range('2000-01-01', periods=max_length, freq=freq).values
    )

    if equal_ends:
        vals_dict['ds'] = (
            np.concatenate(
                [dates[-serie_length:] for serie_length in series_lengths],
            )
        )

    else:
        vals_dict['ds'] = (
            np.concatenate(
                [dates[:serie_length] for serie_length in series_lengths],
            )
        )

    for i in range(n_static_features):
        random.seed(seed)
        static_values = [
            [random.randint(0, 100)] * serie_length for serie_length in series_lengths
        ]
        vals_dict[f'static_{i}'] = np.hstack(static_values)
        if i == 0:
            vals_dict['y'] = vals_dict['y'] * (1 + vals_dict[f'static_{i}'])

    cat_cols = [col for col in vals_dict.keys() if 'static' in col]
    cat_cols.append('unique_id')

    if engine.lower() == 'pandas':
        df = pd.DataFrame(vals_dict)
        df[cat_cols] = df[cat_cols].astype('category')
        df['unique_id'] = df['unique_id'].cat.as_ordered()
        df = df.set_index('unique_id')
        return df
    
    elif engine.lower() == 'polars':
        df = pl.DataFrame(vals_dict)
        df = df.with_columns(pl.col('unique_id').sort())
        for col in cat_cols:
            df = df.with_columns(pl.col(col).cast(str).cast(pl.Categorical))
        return df

    else:
        raise ValueError(f"{engine} is not available.")
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(generate_series, title_level=3)
```

</details>
<details>
<summary>Code</summary>

``` python
from statsforecast.utils import generate_series

synthetic_panel = generate_series(n_series=2)
synthetic_panel.groupby('unique_id').head(4)
```

</details>

# <span style="color:DarkBlue">2. AirPassengers Data </span> {#airpassengers-data}

The classic Box & Jenkins airline data. Monthly totals of international
airline passengers, 1949 to 1960.

It has been used as a reference on several forecasting libraries, since
it is a series that shows clear trends and seasonalities it offers a
nice opportunity to quickly showcase a model’s predictions performance.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
AirPassengers = np.array([112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
                          118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
                          114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
                          162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
                          209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
                          272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
                          302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
                          315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
                          318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
                          348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
                          362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
                          342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
                          417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
                          432.])
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
AirPassengersDF = pd.DataFrame({'unique_id': np.ones(len(AirPassengers)),
                                'ds': pd.date_range(start='1949-01-01',
                                                    periods=len(AirPassengers), freq='M'),
                                'y': AirPassengers})
```

</details>

:::

<details>
<summary>Code</summary>

``` python
from statsforecast.utils import AirPassengersDF

AirPassengersDF.head(12)
```

</details>
<details>
<summary>Code</summary>

``` python
#We are going to plot the ARIMA predictions, and the prediction intervals.
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = AirPassengersDF.set_index('ds')

plot_df[['y']].plot(ax=ax, linewidth=2)
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

</details>

## Model utils {#model-utils}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit
def _repeat_val_seas(season_vals: np.ndarray, h: int, season_length: int):
    out = np.empty(h, np.float32)
    for i in range(h):
        out[i] = season_vals[i % season_length]
    return out

@njit
def _seasonal_naive(
        y: np.ndarray, # time series
        h: int, # forecasting horizon
        fitted: bool, #fitted values
        season_length: int, # season length
    ): 
    if y.size < season_length:
        return {'mean': np.full(h, np.nan, np.float32)}
    n = y.size
    season_vals = np.empty(season_length, np.float32)
    fitted_vals = np.full(y.size, np.nan, np.float32)
    for i in range(season_length):
        s_naive = _naive(y[(i +  n % season_length)::season_length], h=1, fitted=fitted)
        season_vals[i] = s_naive['mean'].item()
        if fitted:
            fitted_vals[(i +  n % season_length)::season_length] = s_naive['fitted']
    out = _repeat_val_seas(season_vals=season_vals, h=h, season_length=season_length)
    fcst = {'mean': out}
    if fitted:
        fcst['fitted'] = fitted_vals[-n:]
    return fcst

@njit
def _repeat_val(val: float, h: int):
    return np.full(h, val, np.float32)

@njit
def _naive(
        y: np.ndarray, # time series
        h: int, # forecasting horizon
        fitted: bool, # fitted values
    ): 
    mean = _repeat_val(val=y[-1], h=h)
    if fitted:
        fitted_vals = np.full(y.size, np.nan, np.float32)
        fitted_vals[1:] = np.roll(y, 1)[1:]
        return {'mean': mean, 'fitted': fitted_vals}
    return {'mean': mean}
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test seasonal naive
y = np.array([0.50187596, 0.40536128, 0.33436676, 0.27868117, 0.25251294,
       0.18961286, 0.07082107, 2.58699709, 3.06466854, 2.25150509,
       1.33027107, 0.73332616, 0.50187596, 0.40536128, 0.33436676,
       0.27868117, 0.25251294, 0.18961286, 0.07082107, 2.58699709,
       3.06466854, 2.25150509, 1.33027107, 0.73332616, 0.50187596,
       0.40536128, 0.33436676, 0.27868117, 0.25251294, 0.18961286,
       0.07082107, 2.58699709, 3.06466854, 2.25150509, 1.33027107,
       0.73332616, 0.50187596, 0.40536128, 0.33436676, 0.27868117,
       0.25251294, 0.18961286, 0.07082107, 2.58699709, 3.06466854,
       2.25150509, 1.33027107, 0.73332616, 0.50187596, 0.40536128,
       0.33436676, 0.27868117, 0.25251294, 0.18961286, 0.07082107,
       2.58699709, 3.06466854, 2.25150509, 1.33027107, 0.73332616,
       0.50187596, 0.40536128, 0.33436676, 0.27868117, 0.25251294,
       0.18961286, 0.07082107, 2.58699709, 3.06466854, 2.25150509,
       1.33027107, 0.73332616, 0.50187596, 0.40536128, 0.33436676,
       0.27868117, 0.25251294, 0.18961286, 0.07082107, 2.58699709,
       3.06466854, 2.25150509, 1.33027107, 0.73332616, 0.50187596,
       0.40536128, 0.33436676, 0.27868117, 0.25251294, 0.18961286,
       0.07082107, 2.58699709, 3.06466854, 2.25150509, 1.33027107,
       0.73332616, 0.50187596, 0.40536128, 0.33436676, 0.27868117,
       0.25251294, 0.18961286, 0.07082107, 2.58699709, 3.06466854,
       2.25150509, 1.33027107, 0.73332616, 0.50187596, 0.40536128,
       0.33436676, 0.27868117, 0.25251294, 0.18961286, 0.07082107,
       2.58699709, 3.06466854, 2.25150509, 1.33027107, 0.73332616,
       0.50187596, 0.40536128, 0.33436676, 0.27868117, 0.25251294,
       0.18961286])
seas_naive_fcst = dict(_seasonal_naive(y=y, h=12, season_length=12, fitted=True))['mean']
np.testing.assert_array_almost_equal(seas_naive_fcst, y[-12:])


y = np.array([0.05293832, 0.10395079, 0.25626143, 0.61529232, 1.08816604,
       0.54493457, 0.43415014, 0.47676606, 5.32806397, 3.00553563,
       0.04473598, 0.04920475, 0.05293832, 0.10395079, 0.25626143,
       0.61529232, 1.08816604, 0.54493457, 0.43415014, 0.47676606,
       5.32806397, 3.00553563, 0.04473598, 0.04920475, 0.05293832,
       0.10395079, 0.25626143, 0.61529232, 1.08816604, 0.54493457,
       0.43415014, 0.47676606, 5.32806397, 3.00553563, 0.04473598,
       0.04920475, 0.05293832, 0.10395079, 0.25626143, 0.61529232,
       1.08816604, 0.54493457, 0.43415014, 0.47676606, 5.32806397,
       3.00553563, 0.04473598, 0.04920475, 0.05293832, 0.10395079,
       0.25626143, 0.61529232, 1.08816604])
seas_naive_fcst = dict(_seasonal_naive(y=y, h=12, season_length=12, fitted=True))['mean']
np.testing.assert_array_almost_equal(seas_naive_fcst, y[-12:])
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
# Functions used for calculating prediction intervals 
def _quantiles(level): 
    level = np.asarray(level)
    z = norm.ppf(0.5+level/200)   
    return z

def _calculate_intervals(out, level, h, sigmah):
    z = _quantiles(np.asarray(level))
    zz = np.repeat(z, h)
    zz = zz.reshape(z.shape[0], h)
    lower = out['mean'] - zz * sigmah
    upper = out['mean'] + zz * sigmah
    pred_int = {**{f'lo-{lv}': lower[i] for i, lv in enumerate(level)}, 
                **{f'hi-{lv}': upper[i] for i, lv in enumerate(level)}}    
    return pred_int

def _calculate_sigma(residuals, n): 
    sigma = np.nansum(residuals ** 2) 
    sigma = sigma / n
    sigma = np.sqrt(sigma)
    return sigma
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
class ConformalIntervals:
    """Class for storing conformal intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method
```

</details>

:::

