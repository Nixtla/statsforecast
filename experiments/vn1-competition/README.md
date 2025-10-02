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
    assert set(ids_forecasts) == set(ids_res), "Some unique_ids are missing"
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
fc = sf.forecast(
    df=df_clean,
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
| 0   | 1st       | 0.4637 |
| 1   | 2nd       | 0.4657 |
| 2   | 3rd       | 0.4758 |
| 3   | 4th       | 0.4774 |
| 4   | 5th       | 0.4808 |
| 9   | Ensemble  | 0.5337 |
| 8   | AutoTheta | 0.5816 |
| 5   | AutoARIMA | 0.6320 |
| 6   | AutoETS   | 0.6452 |
| 7   | CES       | 0.6872 |

</div>

As we can see the median ensemble is much better than models by
themselves.

## 5. Hierarchical approach

The `unique_id` column contains information about the product so the id
is created in the following way: `Client-Warehouse-Product` we’ll build
models based in the Client hierarchy.

``` python
df_client = df_clean.copy()
df_clean[['Client', 'Warehouse', 'Product']] = df_clean['unique_id'].str.split('-', expand=True)
df_client = df_clean.groupby(['Client', 'ds'])['y'].sum().reset_index()
print('There are ', df_client['Client'].nunique(), 'clients in the dataset.')
```

    There are  46 clients in the dataset.

### 5.1 Client level model fitting

We’ll fit the same models for client level predictions.

``` python
sf_client = StatsForecast(
    models=models, 
    freq='W-MON', 
    n_jobs=-1, 
    fallback_model=Naive()
)

fc_client = sf_client.forecast(
    df=df_client, 
    h=13, 
    id_col="Client"
)
```

We also need to identify the obsolete series:

``` python
client_obsolete_series = df_client.groupby("Client").apply(_is_obsolete, days_obsoletes=days_obsoletes)
client_obsolete_ids = client_obsolete_series[client_obsolete_series].index.tolist()
```

We also create the ensemble model for the client level predictions and
set the forecast of obsolete clients to 0:

``` python
fc_client['Ensemble'] = fc_client[['AutoARIMA', 'AutoETS', 'CES', 'AutoTheta']].median(axis=1)

fc_client.loc[fc_client["Client"].isin(client_obsolete_ids), "Ensemble"] = 0
```

### 5.1 Hierarchical Reconciliation with Proportions

We now have two sets of independent forecasts: one at the client level
(aggregated) and one at the product level (granular). These forecasts
are often incoherent—meaning the sum of product-level forecasts for a
client doesn’t match the client-level forecast.

**Proportional reconciliation** resolves this by adjusting the
bottom-level (product) forecasts to sum exactly to the top-level
(client) forecast, while preserving the relative proportions between
products.

**Example:** - Client-level forecast: **120 units** - Product A base
forecast: **50 units** - Product B base forecast: **50 units** - Total
product forecasts: **100 units** (incoherent with client forecast)

Since each product contributes 50% (50/100) of the base forecast,
proportional reconciliation scales both forecasts by the same factor
(120/100 = 1.2):

- Product A reconciled: **60 units** (50 × 1.2)
- Product B reconciled: **60 units** (50 × 1.2)
- Total: **120 units**

This approach leverages the strengths of both aggregation levels: the
stability of aggregate forecasts and the distributional information from
granular forecasts.

![Hierachical reconcilation](../img/hierarchical.png)

Let’s start by creating columns in order to make the join with the
client level version:

``` python
fc[['Client', 'Warehouse', 'Product']] = fc['unique_id'].str.split('-', expand=True)
```

Next we need to have the predictions at the same level so we need to sum
the predicted values in order to have a Client level forecast and
compute the proportions for each product. For this we only use the
`Ensemble` forecast and merge it with the Client level forecast

``` python
total = fc.groupby(['Client', 'ds'])['Ensemble'].sum().reset_index()
total.rename(columns={'Ensemble': 'total_forecasted'}, inplace=True)
total['zero_base_fc'] = np.where(
    total['total_forecasted'] == 0, 
    True, 
    False
)

fc = fc.merge(total, on=['Client', 'ds'], how='left')

fc['fc_proportions'] = fc['Ensemble']/fc['total_forecasted']
fc['fc_proportions'] = fc['fc_proportions'].fillna(0)

fc_client.rename(columns={'Ensemble': 'Ensemble_client', 'unique_id': 'Client'}, inplace=True)

fc = fc.merge(fc_client[['Client', 'ds', 'Ensemble_client']], on=['Client', 'ds'], how='left')
```

Next we have to use this proportions to reconcile with the Client level
forecasted values:

``` python
products_per_client = fc.groupby(['Client', 'ds'])['Product'].nunique().reset_index(name='products_per_client')

fc = fc.merge(products_per_client, on=['Client', 'ds'], how='left')

fc['Ensemble-hierar'] = np.where(
    fc['zero_base_fc'] == False,
    fc['fc_proportions']*fc['Ensemble_client'],
    fc['Ensemble_client']/fc['products_per_client']
)

fc_hierar = fc[['unique_id', 'ds', 'Ensemble-hierar']]
fc_hierar.loc[fc_hierar['Ensemble-hierar'] <= 1e-1, 'Ensemble-hierar'] = 0
```

Let’s proceed to evaluate the results:

``` python
forecasts = forecasts.merge(fc_hierar, on=["unique_id", "ds"], how="left")
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

|     | model           | score  |
|-----|-----------------|--------|
| 0   | 1st             | 0.4637 |
| 1   | 2nd             | 0.4657 |
| 2   | 3rd             | 0.4758 |
| 3   | 4th             | 0.4774 |
| 4   | 5th             | 0.4808 |
| 10  | Ensemble-hierar | 0.4959 |
| 9   | Ensemble        | 0.5337 |
| 8   | AutoTheta       | 0.5816 |
| 5   | AutoARIMA       | 0.6320 |
| 6   | AutoETS         | 0.6452 |
| 7   | CES             | 0.6872 |

</div>

We’ve achivied better results with this hierarchical reconciliation!
