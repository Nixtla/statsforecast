# VN1 Competition


## Introduction to VN1 Competition

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

In the following experiment, we’ll be showcasing how to create forecasts
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

6.  Run the notebook `src/vn1_competition.ipynb`
