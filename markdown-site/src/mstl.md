---
title: MSTL model
---

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from nbdev.showdoc import add_docs, show_doc
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def mstl(
        x: np.ndarray, # time series
        period: Union[int, List[int]], # season length
        blambda: Optional[float] = None, # box-cox transform
        iterate: int = 1, # number of iterations
        s_window: Optional[np.ndarray] = None, # seasonal window
        stl_kwargs: Optional[Dict] = dict(),
    ):
    if s_window is None:
        s_window = 7 + 4 * np.arange(1, 7)
    origx = x
    n = len(x)
    msts = [period] if isinstance(period, int) else period
    iterate = 1
    if x.ndim == 2:
        x = x[:, 0]
    if np.isnan(x).any():
        raise Exception(
            '`mstl` cannot handle missing values. '
            'Please raise an issue to include this feature.'
        ) # we should interpolate here
    if blambda is not None:
        raise Exception(
            '`blambda` not implemented yet. ' 
            'Please rise an issue to include this feature.'
        )
    if msts[0] > 1:
        seas = np.zeros((len(msts), n))
        deseas = np.copy(x)
        if len(s_window) == 1:
            s_window = np.repeat(s_window, len(msts))
        for j in range(iterate):
            for i, seas_ in enumerate(msts, start=0):
                deseas = deseas + seas[i]
                fit = sm.tsa.STL(deseas, period=seas_, seasonal=s_window[i], **stl_kwargs).fit()
                seas[i] = fit.seasonal
                deseas = deseas - seas[i]
        trend = fit.trend
    else:
        try:
            from supersmoother import SuperSmoother
        except ImportError as e:
            print('supersmoother is required for mstl with period=1')
            raise e
        deseas = x
        t = 1 + np.arange(n)
        trend = SuperSmoother().fit(t, x).predict(t)
    deseas[np.isnan(origx)] = np.nan
    remainder = deseas - trend
    output = {'data': origx, 'trend': trend}
    if msts is not None and msts[0] > 1:
        if len(msts) == 1:
            output['seasonal'] = seas[0]
        else:
            for i, seas_ in enumerate(msts, start=0):
                output[f'seasonal{seas_}'] = seas[i]
    output['remainder'] = remainder
    return pd.DataFrame(output)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
x = np.arange(1, 11)
mstl(x, 12)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from statsforecast.utils import AirPassengers as ap
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
decomposition = mstl(ap, 12)
decomposition.plot()
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
decomposition_stl_trend = mstl(ap, 12, stl_kwargs={'trend': 27})
decomposition_stl_trend.plot()
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
decomposition_trend = mstl(ap, 1)
decomposition_trend.plot()
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
df = pd.read_csv(url)
df["Date"] = df["Date"].apply(
    lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
)
df["ds"] = df["Date"] + pd.to_timedelta((df["Period"] - 1) * 30, unit="m")
timeseries = df[["ds", "OperationalLessIndustrial"]]
timeseries.columns = [
    "ds",
    "y",
]  # Rename to OperationalLessIndustrial to y for simplicity.

# Filter for first 149 days of 2012.
start_date = pd.to_datetime("2012-01-01")
end_date = start_date + pd.Timedelta("149D")
mask = (timeseries["ds"] >= start_date) & (timeseries["ds"] < end_date)
timeseries = timeseries[mask]

# Resample to hourly
timeseries = timeseries.set_index("ds").resample("H").sum()
timeseries.head()

# decomposition
decomposition = mstl(timeseries['y'].values, [24, 24 * 7]).tail(24 * 7 * 4)
decomposition.plot()
```

</details>

:::

