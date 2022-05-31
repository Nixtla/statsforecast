import rpy2.robjects as robjects
from rpy2.rinterface_lib import openrlib
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector


forecast = importr('forecast')

def fit_forecast_model(y, model_name, freq, **kwargs):
    pandas2ri.activate()
    rstring = """
     function(y, freq, ...){
         suppressMessages(library(forecast))
         y_ts <- msts(y, seasonal.periods=freq)
         fitted_model<-%s(y_ts, ...)
         fitted_model
     }
    """ % (model_name)
    rfunc = robjects.r(rstring)
    fitted = rfunc(FloatVector(y), freq, **kwargs)
    return fitted

def forecast_object_to_dict(forecast_object):
    """Transform forecast_object into a python dictionary."""
    dict_ = zip(forecast_object.names,
                list(forecast_object))
    dict_ = dict(dict_)
    return dict_

def get_forecast(fitted_model, h):
    """Calculate forecast from a fitted model."""
    y_hat = forecast.forecast(fitted_model, h=h)
    y_hat = forecast_object_to_dict(y_hat)
    y_hat = y_hat['mean']
    return y_hat

def forecast_r(y, h, xreg, freq, model_name):
    model = fit_forecast_model(y, model_name, freq, h=h)
    y_hat = get_forecast(model, h)
    return y_hat

DATASETS = {
    'ili': {'test_size': 193, 'horizons': [24, 36, 48, 60], 'freq': 'W', 'seasonality': 52},
    'ETTm2': {'test_size': 11_520, 'horizons': [96, 192, 336, 720], 'freq': '15T', 'seasonality': 96},
    'Exchange': {'test_size': 1_517, 'horizons': [96, 192, 336, 720], 'freq': 'D', 'seasonality': 7},
    'ECL': {'test_size': 5_260, 'horizons': [96, 192, 336, 720], 'freq': 'H', 'seasonality': 24},
    'weather': {'test_size': 10_539, 'horizons': [96, 192, 336, 720], 'freq': '10T', 'seasonality': 144},
    'traffic': {'test_size': 3_508, 'horizons': [96, 192, 336, 720], 'freq': 'H', 'seasonality': 24},
}
