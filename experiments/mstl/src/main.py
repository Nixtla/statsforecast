from time import time

import pandas as pd
import numpy as np
from datasetsforecast.losses import (
    mae, mape, mase, rmse, smape
)
from fire import Fire
from neuralprophet import NeuralProphet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, SeasonalNaive

def evaluate_performace(y_hist, y_true, models):
    cutoffs = y_true['cutoff'].unique()
    eval_ = []
    for cutoff in cutoffs:
        evaluation = {}
        for model in models:
            evaluation[model] = {}
            for metric in [mase, mae, mape, rmse, smape]:
                metric_name = metric.__name__
                if metric_name == 'mase':
                    evaluation[model][metric_name] = metric(
                        y_true.query('cutoff == @cutoff')['y'].values, 
                        y_true.query('cutoff == @cutoff')[model].values, 
                        y_hist.query('ds <= @cutoff')['y'].values, 
                        seasonality=24
                    )
                else:
                    evaluation[model][metric_name] = metric(
                        y_true.query('cutoff == @cutoff')['y'].values, 
                        y_true.query('cutoff == @cutoff')[model].values
                    )
        eval_cutoff = pd.DataFrame(evaluation).T
        eval_cutoff.insert(0, 'cutoff', cutoff)
        eval_cutoff.index = eval_cutoff.index.rename('model')
        eval_.append(eval_cutoff)
    return pd.concat(eval_)

def experiment():
    df = pd.read_csv('https://raw.githubusercontent.com/jnagura/Energy-consumption-prediction-analysis/master/PJM_Load_hourly.csv')
    df.columns = ['ds', 'y']
    df.insert(0, 'unique_id', 'PJM_Load_hourly')
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    # MSTL model
    mstl = MSTL(
	season_length=[24, 24 * 7], # seasonalities of the time series 
	trend_forecaster=AutoARIMA() # model used to forecast trend
    )
    sf = StatsForecast(
        df=df,
	models=[mstl], 
	freq='H'
    )
    init = time()
    forecasts_cv = sf.cross_validation(h=24, n_windows=7, step_size=24)
    end = time()
    time_mstl = (end - init) / 60
    print(f'MSTL Time: {time_mstl:.2f} minutes')

    # SeasonalNaive model
    sf = StatsForecast(
        df=df,
	models=[SeasonalNaive(season_length=24)], 
	freq='H'
    )
    init = time()
    forecasts_cv_seas = sf.cross_validation(h=24, n_windows=7, step_size=24)
    end = time()
    time_seas = (end - init) / 60
    print(f'SeasonalNaive Time: {time_seas:.2f} minutes')
    forecasts_cv = forecasts_cv.merge(forecasts_cv_seas.drop(columns='y'), how='left', on=['unique_id', 'ds', 'cutoff'])

    cutoffs = forecasts_cv['cutoff'].unique()
    # Prophet model
    forecasts_cv['Prophet'] = None
    time_prophet = 0
    for cutoff in cutoffs:
        df_train = df.query('ds <= @cutoff')
        prophet = Prophet()
        # produce forecasts
        init = time()
        prophet.fit(df_train)
        # produce forecasts
        future = prophet.make_future_dataframe(periods=24, freq='H', include_history=False)
        forecast_prophet = prophet.predict(future)
        end = time()
        assert (forecast_prophet['ds'].values == forecasts_cv.query('cutoff == @cutoff')['ds']).all()
        forecasts_cv.loc[forecasts_cv['cutoff'] == cutoff, 'Prophet'] = forecast_prophet['yhat'].values
        # data wrangling
        time_prophet += (end - init) / 60
    print(f'Prophet Time: {time_prophet:.2f} minutes')
    times = pd.DataFrame({
        'model': ['MSTL', 'SeasonalNaive', 'Prophet'], 
        'time (mins)': [time_mstl, time_seas, time_prophet]
    })

    # NeuralProphet
    forecasts_cv['NeuralProphet'] = None
    time_np = 0
    for cutoff in cutoffs:
        df_train = df.query('ds <= @cutoff')
        neuralprophet = NeuralProphet()
        init = time()
        neuralprophet.fit(df_train.drop(columns='unique_id'))
        future = neuralprophet.make_future_dataframe(df=df_train.drop(columns='unique_id'), periods=24)
        forecast_np = neuralprophet.predict(future)
        end = time()
        assert (forecast_np['ds'].values == forecasts_cv.query('cutoff == @cutoff')['ds']).all()
        forecasts_cv.loc[forecasts_cv['cutoff'] == cutoff, 'NeuralProphet'] = forecast_np['yhat1'].values
        time_np += (end - init) / 60
    print(f'NeuralProphet Time: {time_np:.2f} minutes')
    times = times.append({'model': 'NeuralProphet', 'time (mins)': time_np}, ignore_index=True)
    # Final evalaution
    evaluation = evaluate_performace(df_train, forecasts_cv, models=['MSTL', 'NeuralProphet', 'Prophet', 'SeasonalNaive'])
    print(times)
    print(evaluation)
    print(evaluation.groupby('model').mean(numeric_only=True))


if __name__=="__main__":
    Fire(experiment)
