---
title: Dask
---

export const quartoRawHtml =
[`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>`,`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
</div>`];

> Run StatsForecast distributedly on top of Dask.

StatsForecast works on top of Spark, Dask, and Ray through
[Fugue](https://github.com/fugue-project/fugue/). StatsForecast will
read the input DataFrame and use the corresponding engine. For example,
if the input is a Spark DataFrame, StatsForecast will use the existing
Spark session to run the forecast.

## Installation {#installation}

As long as Dask is installed and configured, StatsForecast will be able
to use it. If executing on a distributed Dask cluster, make use the
`statsforecast` library is installed across all the workers.

## StatsForecast on Pandas {#statsforecast-on-pandas}

Before running on Dask, it’s recommended to test on a smaller Pandas
dataset to make sure everything is working. This example also helps show
the small differences when using Dask.

<details>
<summary>Code</summary>

``` python
from statsforecast.core import StatsForecast
from statsforecast.models import ( 
    AutoARIMA,
    AutoETS,
)
from statsforecast.utils import generate_series

n_series = 4
horizon = 7

series = generate_series(n_series)

sf = StatsForecast(
    models=[AutoETS(season_length=7)],
    freq='D',
)
sf.forecast(df=series, h=horizon).head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

|           | ds         | AutoETS  |
|-----------|------------|----------|
| unique_id |            |          |
| 0         | 2000-08-10 | 5.261609 |
| 0         | 2000-08-11 | 6.196357 |
| 0         | 2000-08-12 | 0.282309 |
| 0         | 2000-08-13 | 1.264195 |
| 0         | 2000-08-14 | 2.262453 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

## Executing on Dask {#executing-on-dask}

To run the forecasts distributed on Dask, just pass in a Dask DataFrame
instead. Instead of having the `unique_id` as an index, it needs to be a
column because Dask handles the index differently.

<details>
<summary>Code</summary>

``` python
import dask.dataframe as dd

# Make unique_id a column
series = series.reset_index()
series['unique_id'] = series['unique_id'].astype(str)

ddf = dd.from_pandas(series, npartitions=4)
```

</details>
<details>
<summary>Code</summary>

``` python
sf.forecast(df=ddf, h=horizon).compute().head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

|     | unique_id | ds         | AutoETS  |
|-----|-----------|------------|----------|
| 0   | 0         | 2000-08-10 | 5.261609 |
| 1   | 0         | 2000-08-11 | 6.196357 |
| 2   | 0         | 2000-08-12 | 0.282309 |
| 3   | 0         | 2000-08-13 | 1.264195 |
| 4   | 0         | 2000-08-14 | 2.262453 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

