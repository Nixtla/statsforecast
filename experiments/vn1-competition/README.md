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

## Load Data and Dependencies

We begin by importing the necessary libraries: - **pandas** and
**numpy** for data manipulation - **StatsForecast** for time series
forecasting with statistical models - **AutoARIMA, AutoETS, AutoCES,
AutoTheta** - automatic model selection algorithms - **Naive** as a
fallback model for problematic series - **fill_gaps** utility for
handling missing timestamps in time series data

``` python
import pandas as pd 
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive
from utilsforecast.preprocessing import fill_gaps
```

### Transform Data to Long Format

The downloaded competition data comes in wide format (one column per
timestamp). For time series modeling with `statsforecast`, we need to
reshape it into long format with three essential columns: -
**`unique_id`**: Identifies each time series (constructed as
`Client-Warehouse-Product`) - **`ds`**: Date stamp (timestamp for each
observation) - **`y`**: Target values to forecast (sales quantities)

This transformation enables efficient processing of thousands of
product-level time series.

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

## Load Top 5 Competition Solutions

To benchmark our approach, we load the forecasts from the competition’s
top 5 performers. This allows us to: 1. Compare `statsforecast` model
performance against winning solutions 2. Understand the competitive
landscape and what accuracy levels are achievable 3. Validate that our
methodology produces competitive results

The competition submissions are stored in wide format CSV files. We load
all five top solutions and merge them into a single dataframe for
comparative evaluation.

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

### Define Competition Evaluation Metric

The VN1 competition uses a custom metric that penalizes both forecast
error magnitude and bias:

**Score = (Sum of Absolute Errors + Absolute Sum of Errors) / Sum of
Actuals**

Lower scores indicate better performance, with the metric equally
weighting accuracy and bias.

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

## Data Preprocessing

### Remove Leading Zeros

Many product-warehouse-client combinations have leading zeros in their
sales history, indicating the product wasn’t available or stocked yet.
For example, `0-1-11000` shows zeros from July 2020 until sales began
later.

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

Including these zeros can: - Bias model estimates of seasonality and
trend - Reduce forecast accuracy by training on non-informative data -
Misrepresent the true demand distribution

We remove leading zeros to train models only on periods when products
were actively sold.

We apply a custom groupby function that identifies the first non-zero
value for each series and discards all preceding observations. This
reduces the dataset from ~2.75M to ~1.62M observations.

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

### Identify Obsolete Products

Some products have extended periods of zero sales at the end of their
history, suggesting they may be discontinued, out of stock, or no longer
carried. For example, product `9-82-9800` shows zeros for the final
months of the training period.

For these obsolete series: - Statistical models may still predict
positive sales based on historical patterns - The correct forecast is
likely zero (product discontinued) - Forecasting non-zero values would
introduce unnecessary error

We identify series with 180 days of consecutive zeros at the end of the
training period and flag them as obsolete.

Examples of obsolete patterns include products with zero sales in recent
history. These require special treatment to avoid spurious forecasts.

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

We flag these obsolete series to override their forecasts with zeros,
preventing statistical models from predicting demand for discontinued
products.

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

## Model Fitting and Base Forecasts

We train four automatic model selection algorithms on the cleaned
data: - **AutoARIMA**: Automatically selects the best ARIMA(p,d,q) model
with seasonal components - **AutoETS**: Chooses optimal
Error-Trend-Seasonality exponential smoothing model - **AutoCES**:
Complex Exponential Smoothing with automatic parameter selection -
**AutoTheta**: Decomposes series into trend and seasonal components
using Theta method

All models use `season_length=52` for weekly seasonality (52 weeks per
year). The `Naive` model serves as a fallback for series where other
models fail to converge.

We instantiate the StatsForecast object with parallel processing
(`n_jobs=-1`) to leverage all available CPU cores, enabling efficient
fitting across thousands of product series.

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

### Model Ensemble

Individual models often have complementary strengths and weaknesses.
Ensembling combines multiple models to: - Reduce variance in
predictions - Mitigate the impact of model-specific biases - Achieve
more robust forecasts than any single model

We use **median ensembling** rather than mean, which is more robust to
outlier predictions from individual models.

The ensemble takes the median of all four model forecasts for each
product-timestamp. We also apply a threshold to set very small forecasts
(\<0.1 units) to zero, reflecting the discrete nature of sales.

``` python
fc['Ensemble'] = fc[['AutoARIMA', 'AutoETS', 'CES', 'AutoTheta']].median(axis=1)
fc.loc[fc['Ensemble'] <= 1e-1, 'Ensemble'] = 0
```

For products identified as obsolete during preprocessing, we override
all model forecasts (including the ensemble) with zero, regardless of
what the statistical models predict.

``` python
fc.loc[fc["unique_id"].isin(obsolete_ids), "Ensemble"] = 0
```

### Evaluate Base Model Performance

We compare our forecasts against the competition’s top 5 solutions using
the custom VN1 metric.

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

**Key Result**: The median ensemble (score: 0.5337) significantly
outperforms individual models, with improvements of 8-22% over
single-model approaches. However, there’s still a gap to the top
competition solutions (~0.46-0.48 range), suggesting potential for
further refinement.

## Hierarchical Reconciliation Approach

The data has a natural hierarchy: **Client → Warehouse → Product**. Each
`unique_id` encodes this structure as `Client-Warehouse-Product`.

**Key Insight**: Forecasts at different hierarchical levels contain
complementary information: - **Aggregate (client) level**: More stable,
less noisy, captures overall demand trends - **Bottom (product) level**:
Captures product-specific patterns but more volatile

Hierarchical reconciliation combines both levels to produce coherent
forecasts where: 1. Individual product forecasts are adjusted to match
aggregate forecasts 2. The relative distribution among products is
preserved 3. We leverage the stability of aggregate forecasts with the
granularity of product forecasts

We extract the hierarchical structure from the `unique_id` field and
aggregate sales to the client level. The dataset contains 46 unique
clients, each with multiple warehouse-product combinations.

``` python
df_client = df_clean.copy()
df_clean[['Client', 'Warehouse', 'Product']] = df_clean['unique_id'].str.split('-', expand=True)
df_client = df_clean.groupby(['Client', 'ds'])['y'].sum().reset_index()
print('There are ', df_client['Client'].nunique(), 'clients in the dataset.')
```

    There are  46 clients in the dataset.

### Fit Models at Client Level

We train the same set of statistical models (AutoARIMA, AutoETS,
AutoCES, AutoTheta) on aggregated client-level time series. These client
forecasts will serve as the “target totals” for hierarchical
reconciliation.

Client-level series have less volatility than individual products, often
yielding more reliable forecasts.

We use identical model specifications as the product-level forecasts to
ensure consistency in methodology across hierarchy levels.

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

We also check for obsolete clients (those with 180 days of zero sales)
to apply the same zero-forecast override logic at the aggregate level.

``` python
client_obsolete_series = df_client.groupby("Client").apply(_is_obsolete, days_obsoletes=days_obsoletes)
client_obsolete_ids = client_obsolete_series[client_obsolete_series].index.tolist()
```

We create a client-level ensemble using the same median approach and
override forecasts for obsolete clients.

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

We extract the hierarchical components from the `unique_id` to enable
merging product-level and client-level forecasts.

``` python
fc[['Client', 'Warehouse', 'Product']] = fc['unique_id'].str.split('-', expand=True)
```

### Calculate Product Proportions

To reconcile forecasts, we need to understand each product’s
contribution to its client’s total forecast:

1.  **Aggregate product forecasts** to client level (sum all products
    per client)
2.  **Calculate proportions**: Each product’s share of the client total
3.  **Handle edge cases**: When base forecast totals are zero,
    distribute equally among products

These proportions capture the expected distribution of demand across
products within each client.

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

### Apply Proportional Reconciliation

We reconcile the product-level forecasts using:

**Reconciled Product Forecast = Product Proportion × Client-Level
Forecast**

This ensures: - Sum of reconciled product forecasts = Client forecast
(coherence) - Relative distribution among products preserved (from base
forecasts) - Leverages the typically more accurate client-level
aggregated forecast

**Special case**: When the base product forecast sum is zero, we
distribute the client forecast equally across all products for that
client.

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

### Evaluate Hierarchical Reconciliation

We compare the hierarchically reconciled forecasts against both the base
ensemble and the competition solutions.

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

**Significant Improvement**: The hierarchical ensemble achieves a score
of 0.4959, representing: - **7.1% improvement** over the base ensemble
(0.5337) - **Competitive performance**, approaching the 5th place
solution (0.4808) - Demonstrates the value of leveraging hierarchical
structure in the data

This shows that hierarchical reconciliation successfully combines the
stability of aggregate forecasts with product-level granularity, closing
the gap toward competition-winning performance.
