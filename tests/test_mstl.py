import numpy as np
import pandas as pd
from statsforecast.mstl import mstl
from statsforecast.utils import AirPassengers as ap

x = np.arange(1, 11)
mstl(x, 12)

decomposition = mstl(ap, 12)
decomposition.plot()
decomposition_stl_trend = mstl(ap, 12, stl_kwargs={"trend": 27})
decomposition_stl_trend.plot()
decomposition_trend = mstl(ap, 1)
decomposition_trend.plot()
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
decomposition = mstl(timeseries["y"].values, [24, 24 * 7]).tail(24 * 7 * 4)
decomposition.plot()
