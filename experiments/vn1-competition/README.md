# VN1 Competition


## Introduction

The VN1 Forecasting Accuracy Challenge tasked participants with
forecasting future sales using historical sales and pricing data. The
goal was to develop robust predictive models capable of anticipating
sales trends for various products across different clients and
warehouses. Submissions were evaluated based on their accuracy and bias
against actual sales figures.

The competition was structured into two phases:

- **Phase 1** (September 12 - October 3, 2024): Participants used the
  provided Phase 0 sales data to predict sales for Phase 1. This phase
  lasted three weeks and featured live leaderboard updates to track
  participant progress.
- **Phase 2** (October 3 - October 17, 2024): Participants utilized both
  Phase 0 and Phase 1 data to predict sales for Phase 2. Unlike Phase 1,
  there were no leaderboard updates during this phase until the
  competition concluded.

In the following notebook, we’ll be showcasing how to create forecasts
with ETS, ARIMA, CES and Theta models from `statsforecast` as well as
using an ensemble of this models and a hierarchical reconciliation.

## Setting up with uv

To set up the environment using `uv`, follow these steps:

1.  Install `uv` if you haven’t already:

``` bash
pip install uv
```

2.  Navigate to the `vn1-competition` directory:

``` bash
cd experiments/vn1-competition
```

3.  Create a virtual environment and install dependencies using uv:

``` bash
uv sync
```

4.  Activate the virtual environment:

``` bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

5.  Download data

``` bash
make download_data
```

## Load data

``` python
import pandas as pd 
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive
from utilsforecast.preprocessing import fill_gaps
```

    /app/.venv/lib/python3.12/site-packages/fs/__init__.py:4: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      __import__("pkg_resources").declare_namespace(__name__)  # type: ignore

The downloaded data is in wide format so we transform the data in order
to be used by `statsforecast`. This imply a long dataframe with columns
`unique_id` denoting the time serie identifier, `ds` the date stamp and
`y` the values to be forecasted.

``` python
def read_and_prepare_data(file_path: str, value_name: str = "y") -> pd.DataFrame:
    """Reads data in wide format, and returns it in long format with columns `unique_id`, `ds`, `y`"""
    df = pd.read_csv(file_path)
    uid_cols = ["Client", "Warehouse", "Product"]
    df["unique_id"] = df[uid_cols].astype(str).agg("-".join, axis=1)
    df = df.drop(uid_cols, axis=1)
    df = df.melt(id_vars=["unique_id"], var_name="ds", value_name=value_name)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    return df
```

``` python
df0 = read_and_prepare_data("../data/phase_0_sales.csv")
df1 = read_and_prepare_data("../data/phase_1_sales.csv")

df = pd.concat([df0, df1], ignore_index=True)
df = df.sort_values(by=["unique_id", "ds"])
test_df = read_and_prepare_data("../data/phase_2_sales.csv")
```

## 2. Load Top 5 Solutions

In order to compare the result that we get with `statsforecast` we load
the predictions for top 5 competitors:

``` python
def get_competition_forecasts() -> pd.DataFrame|None:
    """Reads all competition forecasts and returns it in long format with columns `unique_id`, `ds`, `y`"""
    fcst_df: pd.DataFrame | None = None
    for place in ["1st", "2nd", "3rd", "4th", "5th"]:
        fcst_df_place = read_and_prepare_data(
            f"../data/solution_{place}_place.csv", place
        )
        if fcst_df is None:
            fcst_df = fcst_df_place
        else:
            fcst_df = fcst_df.merge(
                fcst_df_place,
                on=["unique_id", "ds"],
                how="left",
            )
    return fcst_df
```

``` python
solutions = get_competition_forecasts()
```

To evaluate the predictions from any prediction we provide the following
function:

``` python
def vn1_competition_evaluation(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Computes competition evaluation scores"""
    actual = read_and_prepare_data("../data/phase_2_sales.csv")
    res = actual[["unique_id", "ds", "y"]].merge(
        forecasts, on=["unique_id", "ds"], how="left"
    )
    ids_forecasts = forecasts["unique_id"].unique()
    ids_res = res["unique_id"].unique()
    res = res.query("unique_id in @ ids_forecasts")
    #assert set(ids_forecasts) == set(ids_res), "Some unique_ids are missing"
    scores = {}
    for model in [col for col in forecasts.columns if col not in ["unique_id", "ds"]]:
        abs_err = np.nansum(np.abs(res[model] - res["y"]))
        err = np.nansum(res[model] - res["y"])
        score = abs_err + abs(err)
        score = score / res["y"].sum()
        scores[model] = round(score, 4)
    score_df = pd.DataFrame(list(scores.items()), columns=["model", "score"])
    score_df = score_df.sort_values(by="score")
    return score_df
```

## 3. Data Processing

### 3.1. Remove leading zeros

There are some `unique_id` that have starting values in `0` meaning that
the product wasn’t present at the time, for example:

``` python
df.query("unique_id == '0-1-11000'")
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|         | unique_id | ds         | y    |
|---------|-----------|------------|------|
| 170     | 0-1-11000 | 2020-07-06 | 0.0  |
| 171     | 0-1-11000 | 2020-07-13 | 0.0  |
| 172     | 0-1-11000 | 2020-07-20 | 0.0  |
| 173     | 0-1-11000 | 2020-07-27 | 0.0  |
| 174     | 0-1-11000 | 2020-08-03 | 0.0  |
| ...     | ...       | ...        | ...  |
| 2559031 | 0-1-11000 | 2023-12-04 | 5.0  |
| 2559032 | 0-1-11000 | 2023-12-11 | 20.0 |
| 2559033 | 0-1-11000 | 2023-12-18 | 10.0 |
| 2559034 | 0-1-11000 | 2023-12-25 | 6.0  |
| 2559035 | 0-1-11000 | 2024-01-01 | 15.0 |

<p>183 rows × 3 columns</p>
</div>

We’ll remove the leading zeros with the following function:

``` python
def _remove_leading_zeros(group): 
    """
    Removes leading zeros from series 
    """
    first_non_zero_index = group['y'].ne(0).idxmax()
    return group.loc[first_non_zero_index:]

df_clean = df.groupby("unique_id").apply(_remove_leading_zeros).reset_index(drop=True)

df_clean.shape, df.shape
```

    ((1615437, 3), (2754699, 3))

### 3.2. Identify obsolete series

There are some products that for a long period haven’t been buyed such
as:

``` python
df_clean.query("unique_id == '9-82-9800'").tail()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|         | unique_id | ds         | y   |
|---------|-----------|------------|-----|
| 1615250 | 9-82-9800 | 2023-12-04 | 0.0 |
| 1615251 | 9-82-9800 | 2023-12-11 | 0.0 |
| 1615252 | 9-82-9800 | 2023-12-18 | 0.0 |
| 1615253 | 9-82-9800 | 2023-12-25 | 0.0 |
| 1615254 | 9-82-9800 | 2024-01-01 | 0.0 |

</div>

We want to identify them in order to predict 0 demand in this products

``` python
def _is_obsolete(group, days_obsoletes):
    """
    Identify obsolete series
    """
    last_date = group["ds"].max()
    cutoff_date = last_date - pd.Timedelta(days=days_obsoletes)
    recent_data = group.query("ds >= @cutoff_date")
    return (recent_data["y"] == 0).all()

days_obsoletes=180 # context-dependent 
obsolete_series = df_clean.groupby("unique_id").apply(_is_obsolete, days_obsoletes=days_obsoletes)
obsolete_ids = obsolete_series[obsolete_series].index.tolist()
```

## 4. Model fitting

Now we proceed to fit the `AutoARIMA`, `AutoETS`, `AutoCES` and
`AutoTheta` models to all products.

``` python
models = [
    AutoARIMA(season_length=52), 
    AutoETS(season_length=52), 
    AutoCES(season_length=52), 
    AutoTheta(season_length=52)
]

sf = StatsForecast(
    models=models, 
    freq='W-MON', 
    n_jobs=-1, 
    fallback_model=Naive()
)
```

``` python
first = df_clean.unique_id.unique()[0:2]
```

``` python
fc = sf.forecast(
    df=df_clean.query("unique_id in @first"), 
    h=13
)
```

### 4.1 Model ensembling

We now proceed to ensemble the models with the median.

``` python
fc['Ensemble'] = fc[['AutoARIMA', 'AutoETS', 'CES', 'AutoTheta']].median(axis=1)
fc.loc[fc['Ensemble'] <= 1e-1, 'Ensemble'] = 0
```

For obsolete series we provide 0 prediction.

``` python
fc.loc[fc["unique_id"].isin(obsolete_ids), "Ensemble"] = 0
```

### 4.2 Evaluate results

Now we proceed to evaluate the results from the predictions.

``` python
forecasts = solutions.merge(fc, on=["unique_id", "ds"], how="inner")
vn1_competition_evaluation(forecasts)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | model     | score  |
|-----|-----------|--------|
| 3   | 4th       | 1.4175 |
| 7   | CES       | 1.5778 |
| 2   | 3rd       | 1.5933 |
| 8   | AutoTheta | 1.6036 |
| 9   | Ensemble  | 1.6961 |
| 5   | AutoARIMA | 1.7497 |
| 6   | AutoETS   | 1.8290 |
| 0   | 1st       | 1.8337 |
| 1   | 2nd       | 1.8602 |
| 4   | 5th       | 2.4011 |

</div>

As we can see the median ensemble is much better than models by
themselves.
