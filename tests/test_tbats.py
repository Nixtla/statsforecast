import matplotlib.pyplot as plt
import pandas as pd

from statsforecast.utils import AirPassengers as ap
y = ap
seasonal_periods = np.array([12])
# Default parameters 
use_boxcox = None
bc_lower_bound = 0
bc_upper_bound = 1
use_trend = None
use_damped_trend = None
use_arma_errors = True
mod = tbats_selection(y, seasonal_periods, use_boxcox, bc_lower_bound, bc_upper_bound, use_trend, use_damped_trend, use_arma_errors) 
# Values in R
print(mod['aic']) # 1397.015
print(mod['k_vector']) # 5
print(mod['description']) # use_boxcox = TRUE, use_trend = TRUE, use_damped_trend = FALSE, use_arma_errors = FALSE
fitted_trans = mod['fitted'].ravel()
if mod['BoxCox_lambda'] is not None:
    fitted_trans = inv_boxcox(fitted_trans, mod['BoxCox_lambda'])    
h = 24
fcst = tbats_forecast(mod, h)
forecast = fcst['mean']
if mod['BoxCox_lambda'] is not None:
    forecast = inv_boxcox(forecast,  mod['BoxCox_lambda'])
fig, ax = plt.subplots(1, 1, figsize = (20,7))
plt.plot(np.arange(0, len(y)), y, color='black', label='original')
plt.plot(np.arange(0, len(y)), fitted_trans, color='blue', label = "fitted")
plt.plot(np.arange(len(y), len(y)+h), forecast, '.-', color = 'green', label = 'fcst')
plt.legend()
