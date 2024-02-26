from time import time
import pandas as pd
import numpy as np
from tbats import TBATS as TBATSPY

df = pd.read_csv(
        "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv"
    )
df.columns = ["ds", "y"]
df.insert(0, "unique_id", "PJM_Load_hourly")
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
df = df[:-24] # remove test set 
y = df["y"].values 

estimator = TBATSPY(seasonal_periods=[24, 24 * 7])
time_tbatspy = 0
init = time()
fitted_model = estimator.fit(y)
forecast_tbatspy = fitted_model.forecast(steps=24)
end = time()

date_range = pd.date_range(df.iloc[-1]["ds"], periods=25, freq="H")
date_range = date_range[1:]
fcst_df = pd.DataFrame(
    {
        "unique_id": "PJM_Load_hourly",
        "ds": date_range,
        "y": df.tail(24)["y"],
        "tbats_py": forecast_tbatspy,
    }

)
fcst_df.to_csv("data/tbats-py.csv", index=False)

time_tbatspy = (end - init) / 60
time_df = pd.DataFrame(
    {
        "model": "tbats_py",
        "time": time_tbatspy,
    },
    index=[0],

)
time_df.to_csv("data/tbats-py-time.csv", index=False)
