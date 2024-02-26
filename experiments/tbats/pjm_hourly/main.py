from functools import partial
from time import time

import numpy as np
import pandas as pd
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoTBATS
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse

def evaluate_forecasts(df: pd.DataFrame, forecasts_cv: pd.DataFrame):
    evaluation_df = []
    cutoffs = forecasts_cv["cutoff"].unique()
    for cutoff in cutoffs:
        eval_cutoff_df = evaluate(
            forecasts_cv.query("cutoff == @cutoff").drop("cutoff", axis=1),
            train_df=df.query("ds <= @cutoff"),
            metrics=[mae, rmse],
        )
        eval_cutoff_df["cutoff"] = cutoff
        evaluation_df.append(eval_cutoff_df)
    evaluation_df = pd.concat(evaluation_df)
    return evaluation_df

def experiment():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv"
    )
    df.columns = ["ds", "y"]
    df.insert(0, "unique_id", "PJM_Load_hourly")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True) 

    # Seasonal Naive model
    sf = StatsForecast(models=[SeasonalNaive(season_length=24)], freq="H")
    init = time()
    forecasts_cv_seas = sf.cross_validation(df=df, h=24, n_windows=5, step_size=24)
    end = time()
    time_seas = (end - init) / 60
    print(f"SeasonalNaive Time: {time_seas:.2f} minutes")
    
    # TBATS model
    sf = StatsForecast(models=[AutoTBATS(seasonal_periods=[24, 24*7])], freq="H")
    init = time()
    forecasts_cv = sf.cross_validation(df=df, h=24, n_windows=5, step_size=24)
    end = time()
    time_tbats = (end - init) / 60
    print(f"AutoTBATS Time: {time_tbats:.2f} minutes")

    forecasts_cv = forecasts_cv.merge(
        forecasts_cv_seas.drop(columns="y"),
        how="left",
        on=["unique_id", "ds", "cutoff"],
    )
    
    cutoffs = forecasts_cv["cutoff"].unique()
    
    # Prophet model
    forecasts_cv["Prophet"] = None
    time_prophet = 0
    for cutoff in cutoffs:
        df_train = df.query("ds <= @cutoff")
        prophet = Prophet()
        init = time()
        prophet.fit(df_train)
        future = prophet.make_future_dataframe(
            periods=24, freq="H", include_history=False
        )
        forecast_prophet = prophet.predict(future)
        end = time()
        assert (
            forecast_prophet["ds"].values
            == forecasts_cv.query("cutoff == @cutoff")["ds"]
        ).all()
        forecasts_cv.loc[
            forecasts_cv["cutoff"] == cutoff, "Prophet"
        ] = forecast_prophet["yhat"].values
        time_prophet += (end - init) / 60
    print(f"Prophet Time: {time_prophet:.2f} minutes")

    # TBATS-R and TBATS-PY models
    time_tbats_r = pd.read_csv("data/tbats-r-time.csv")["time"].item()
    time_tbats_py = pd.read_csv("data/tbats-py-time.csv")["time"].item()

    times = pd.DataFrame(
        {
            "model": ["AutoTBATS", "TBATS-R", "TBATS-PY", "SeasonalNaive", "Prophet"], 
            "time (mins)": [time_tbats, time_tbats_r, time_tbats_py, time_seas, time_prophet], 
        }
    )
    
    tbats_r_fcst_df = pd.read_csv("data/tbats-r.csv")
    np.testing.assert_array_equal(
        tbats_r_fcst_df["y"].values,
        forecasts_cv_seas["y"].values,
    )

    # tbats_py_fcst_df = pd.read_csv("data/tbats-py.csv")
    # np.testing.assert_array_equal(
    #     tbats_py_fcst_df["y"].values,
    #     forecasts_cv_seas["y"].values,
    # )

    forecasts_cv["TBATS-R"] = tbats_r_fcst_df["tbats_r"].values
    # forecasts_cv["TBATS-PY"] = tbats_py_fcst_df["tbats_py"].values

    # Final evalaution
    evaluation_df = evaluate_forecasts(df, forecasts_cv.reset_index())
    mean_evaluation_df = evaluation_df.groupby("metric")[["AutoTBATS", "TBATS-R", "SeasonalNaive", "Prophet"]].mean()
    
    print(times)
    print(evaluation_df)
    print(mean_evaluation_df)


if __name__ == "__main__":
    experiment()