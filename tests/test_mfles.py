import pandas as pd
from statsforecast.mfles import MFLES

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
mfles = MFLES()
fitted = mfles.fit(y=timeseries.y.values, seasonal_period=[24, 24 * 7])
predicted = mfles.predict(forecast_horizon=24)
mfles = MFLES()
opt_params = mfles.optimize(timeseries.y.values,
                            seasonal_period=[24, 24 * 7],
                            n_steps=3,
                            test_size=24,
                            step_size=24)
fitted = mfles.fit(y=timeseries.y.values, seasonal_period=[24, 24 * 7])
predicted = mfles.predict(forecast_horizon=24)
print(opt_params)
