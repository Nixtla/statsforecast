%load_ext autoreload
%autoreload 2
from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from fastcore.test import test_eq, test_close, test_fail
from nbdev.showdoc import add_docs, show_doc

from statsforecast.garch import generate_garch_data
from statsforecast.utils import AirPassengers as ap
def _plot_insample_pi(fcst):
    
    fig, ax = plt.subplots(1, 1, figsize = (20,7))

    sdate = date(1949,1,1) # start date 
    edate = date(1961,1,1) # end date
    dates = pd.date_range(sdate,edate-timedelta(days=1),freq='m')

    df = pd.DataFrame({'dates': dates,
                       'actual': ap,
                       'fitted': fcst['fitted'],
                       'fitted_lo_80': fcst['fitted-lo-80'], 
                       'fitted_lo_95': fcst['fitted-lo-95'], 
                       'fitted_hi_80': fcst['fitted-hi-80'],
                       'fitted_hi_95': fcst['fitted-hi-95']})
    plt.plot(df.dates, df.actual, color='firebrick', label='Actual value', linewidth=3)
    plt.plot(df.dates, df.fitted, color='navy', label='Fitted values', linewidth=3)
    plt.plot(df.dates, df.fitted_lo_80, color='darkorange', label='fitted-lo-80', linewidth=3)
    plt.plot(df.dates, df.fitted_lo_95, color='deepskyblue', label='fitted-lo-95', linewidth=3)
    plt.plot(df.dates, df.fitted_hi_80, color='darkorange', label='fitted-hi-80', linewidth=3)
    plt.plot(df.dates, df.fitted_hi_95, color='deepskyblue', label='fitted-hi-95', linewidth=3)
    plt.fill_between(df.dates, df.fitted_lo_95, df.fitted_hi_95, color = 'deepskyblue', alpha = 0.2)
    plt.fill_between(df.dates, df.fitted_lo_80, df.fitted_hi_80, color = 'darkorange', alpha = 0.3)
    plt.legend()
    
def _plot_fcst(fcst): 
    
    fig, ax = plt.subplots(1, 1, figsize = (20,7))
    plt.plot(np.arange(0, len(ap)), ap)
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst['mean'], label='mean')
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst['lo-95'], color = 'r', label='lo-95')
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst['hi-95'], color = 'r', label='hi-95')
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst['lo-80'], color = 'g', label='lo-80')
    plt.plot(np.arange(len(ap), len(ap) + 13), fcst['hi-80'], color = 'g', label='hi-80')
    plt.legend()
# test conformity scores
class ZeroModel(_TS):
    
    def __init__(self, prediction_intervals: ConformalIntervals = None):
        self.prediction_intervals = prediction_intervals
        self.alias = 'SumAhead'
    
    def forecast(self, y, h, X=None, X_future=None, fitted=False, level=None):
        res = {'mean': np.zeros(h)}
        if self.prediction_intervals is not None and level is not None:
            cs = self._conformity_scores(y, X)
            res = self._conformal_method(fcst=res, cs=cs, level=level)
        return res
    
    def fit(self, y, X):
        return self
    
    def predict(self, h, X=None, level=None):
        res = {'mean': np.zeros(h)}
        return res
conf_intervals = ConformalIntervals(h=12, n_windows=10)
expected_cs = np.full((conf_intervals.n_windows, conf_intervals.h), np.nan)
cs_info = ap[-conf_intervals.h*conf_intervals.n_windows:]
for i in range(conf_intervals.n_windows):
    expected_cs[i] = cs_info[i * conf_intervals.h:(i+1) * conf_intervals.h]
current_cs = ZeroModel(conf_intervals)._conformity_scores(ap)
test_eq(expected_cs, current_cs)
zero_model = ZeroModel(conf_intervals)
fcst_conformal = zero_model.forecast(ap, h=12, level=[80, 90])
test_eq(list(fcst_conformal.keys()), ['mean', 'lo-90', 'lo-80', 'hi-80', 'hi-90'])
def test_class(
    cls_, x, h, skip_insample=False, level=None, test_forward=False, X=None, X_future=None
):
    cls_ = cls_.fit(x, X=X)
    fcst_cls = cls_.predict(h=h, X=X_future)
    test_eq(len(fcst_cls['mean']), h)
    # test fit + predict equals forecast
    test_eq(
        cls_.forecast(y=x, h=h, X=X, X_future=X_future)['mean'],
        fcst_cls['mean']
    )
    if not skip_insample:
        test_eq(len(cls_.predict_in_sample()['fitted']), len(x))
        assert isinstance(cls_.predict_in_sample()['fitted'], np.ndarray)
        np.testing.assert_array_equal(
            cls_.forecast(y=x, h=h, X=X, X_future=X_future, fitted=True)['fitted'],
            cls_.predict_in_sample()['fitted'], 
        )
        if test_forward:
            np.testing.assert_array_equal(
                cls_.forward(y=x, h=h, X=X, X_future=X_future, fitted=True)['fitted'],
                cls_.predict_in_sample()['fitted'], 
            )

    if test_forward:
        try:
            pd.testing.assert_frame_equal(
                pd.DataFrame(cls_.predict(h=h, X=X_future)),
                pd.DataFrame(cls_.forward(y=x, X=X, X_future=X_future, h=h)),
            )
        except AssertionError:
            raise Exception('predict and forward methods are not equal')
    
    if level is not None:
        fcst_cls = pd.DataFrame(cls_.predict(h=h, X=X_future, level=level))
        fcst_forecast = pd.DataFrame(cls_.forecast(y=x, h=h, X=X, X_future=X_future, level=level))
        try:
            pd.testing.assert_frame_equal(fcst_cls, fcst_forecast)
        except AssertionError:
            raise Exception('predict and forecast methods are not equal with levels')
            
        if test_forward:
            try:
                pd.testing.assert_frame_equal(
                    pd.DataFrame(cls_.predict(h=h, X=X_future, level=level)),
                    pd.DataFrame(cls_.forward(y=x, h=h, X=X, X_future=X_future, level=level))
                )
            except AssertionError:
                raise Exception('predict and forward methods are not equal with levels')
        
        if not skip_insample:
            fcst_cls = pd.DataFrame(cls_.predict_in_sample(level=level))
            fcst_forecast = cls_.forecast(y=x, h=h, X=X, X_future=X_future, level=level, fitted=True)
            fcst_forecast = pd.DataFrame({key: val for key, val in fcst_forecast.items() if 'fitted' in key})
            try:
                pd.testing.assert_frame_equal(fcst_cls, fcst_forecast)
            except AssertionError:
                raise Exception(
                    'predict and forecast methods are not equal with ' 
                    'levels for fitted values '
                )
            if test_forward:
                fcst_forward = cls_.forecast(y=x, h=h, X=X, X_future=X_future, level=level, fitted=True)
                fcst_forward = pd.DataFrame({key: val for key, val in fcst_forward.items() if 'fitted' in key})
                try:
                    pd.testing.assert_frame_equal(fcst_cls, fcst_forward)
                except AssertionError:
                    raise Exception(
                        'predict and forward methods are not equal with ' 
                        'levels for fitted values '
                    )

def _test_fitted_sparse(model_factory):
    y1 = np.array([2, 5, 0, 1, 3, 0, 1, 1, 0], dtype=np.float64)
    y2 = np.array([0, 0, 1, 0, 0, 7, 1, 0, 1], dtype=np.float64)
    y3 = np.array([0, 0, 1, 0, 0, 7, 1, 0, 0], dtype=np.float64)
    y4 = np.zeros(9, dtype=np.float64)
    for y in [y1, y2, y3, y4]:
        expected_fitted = np.hstack(
            [
                model_factory().forecast(y=y[:i + 1], h=1)['mean']
                for i in range(y.size - 1)]
        )
        np.testing.assert_allclose(
            model_factory().forecast(y=y, h=1, fitted=True)['fitted'],
            np.append(np.nan, expected_fitted),
            atol=1e-6,
        )
arima = AutoARIMA(season_length=12) 
test_class(arima, x=ap, h=12, level=[90, 80], test_forward=True)
fcst_arima = arima.forecast(ap, 13, None, None, (80,95), True)
_plot_insample_pi(fcst_arima)
# test conformal prediction
arima_c = AutoARIMA(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(arima_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_arima_c = arima_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_arima_c["mean"],
    fcst_arima["mean"],
)
_plot_fcst(fcst_arima_c)
#test alias argument
test_eq(
    repr(AutoARIMA()),
    'AutoARIMA'
)
test_eq(
    repr(AutoARIMA(alias='AutoARIMA_seasonality')),
    'AutoARIMA_seasonality'
)
_plot_fcst(fcst_arima)
show_doc(AutoARIMA, title_level=3)
show_doc(AutoARIMA.fit, title_level=3)
show_doc(AutoARIMA.predict, title_level=3)
show_doc(AutoARIMA.predict_in_sample, title_level=3)
show_doc(AutoARIMA.forecast, title_level=3)
show_doc(AutoARIMA.forward, title_level=3)
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengers as ap
# AutoARIMA's usage example
arima = AutoARIMA(season_length=4)
arima = arima.fit(y=ap)
y_hat_dict = arima.predict(h=4, level=[80])
y_hat_dict
autoets = AutoETS(season_length=12)
test_class(autoets, x=ap, h=12, level=[90, 80], test_forward=True)
fcst_ets = autoets.forecast(ap, 13, None, None, (80,95), True)
_plot_insample_pi(fcst_ets)
_plot_fcst(fcst_ets)
#test alias argument
test_eq(
    repr(AutoETS()),
    'AutoETS'
)
test_eq(
    repr(AutoETS(alias='AutoETS_custom')),
    'AutoETS_custom'
)
autoets = AutoETS(season_length=12, model='AAA')
test_class(autoets, x=ap, h=12, level=[90, 80])
fcst_ets = autoets.forecast(ap, 13, None, None, (80,95), True)
_plot_insample_pi(fcst_ets)
# test conformal prediction
autoets_c = AutoETS(season_length=12, model='AAA', prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(autoets_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_ets_c = autoets_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(fcst_ets_c['mean'],
    fcst_ets['mean'])
_plot_fcst(fcst_ets_c)
# Test whether forecast and fit-predict generate the same result 
models = ['ANNN', 'AANN', 'ANAN', 'AAAN', 'AAND', 'AAAD', # class 1 
          'MNNN', 'MANN', 'MAND', 'MNAN', 'MAAN', 'MAAD', # class 2 
          'MNMN', 'MAMN', 'MAMD'] # class 3 

for k in range(0,len(models)): 
    mod = models[k][0:3]
    damped_val = models[k][-1]
    if damped_val == 'N': 
        damped = False
    else: 
        damped = True
        
    ets = AutoETS(season_length=12, model=mod, damped=damped) 
    test_class(ets, x=ap, h=13, level=[90, 80], test_forward=True)
show_doc(AutoETS, title_level=3)
show_doc(AutoETS.fit, title_level=3)
show_doc(AutoETS.predict, title_level=3)
show_doc(AutoETS.predict_in_sample, title_level=3)
show_doc(AutoETS.forecast, title_level=3)
show_doc(AutoETS.forward, title_level=3)
from statsforecast.models import AutoETS
from statsforecast.utils import AirPassengers as ap

# AutoETS' usage example
# Multiplicative trend, optimal error and seasonality
autoets = AutoETS(model='ZMZ', season_length=4)
autoets = autoets.fit(y=ap)
y_hat_dict = autoets.predict(h=4)
y_hat_dict
autoces = AutoCES(season_length=12)
fcst_ces = autoces.forecast(ap, 13, None, None, (80,95), True)
_plot_insample_pi(fcst_ces)
# test conformal prediction
autoces_c = AutoCES(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(autoces_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_ces_c = autoces_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_ces["mean"],
    fcst_ces_c["mean"]
)
_plot_fcst(fcst_ces)
_plot_fcst(fcst_ces_c)
fit = autoces.fit(ap) 
fcst = fit.predict(13, None, (80,95))
_plot_fcst(fit.predict(13, None, (80,95)))

values = ['mean', 'lo-80', 'lo-95', 'hi-80', 'hi-95']
for k in range(0, len(values)): 
    np.testing.assert_equal(
    fcst_ces[values[k]], 
    fcst[values[k]]
)
pi_insample = fit.predict_in_sample((80,95))
_plot_insample_pi(pi_insample)

values = ['fitted', 'fitted-lo-80', 'fitted-lo-95', 'fitted-hi-80', 'fitted-hi-95']
for k in range(0, len(values)): 
    np.testing.assert_equal(
    fcst_ces[values[k]], 
    pi_insample[values[k]]
)
ces = AutoCES(season_length=12)
test_class(ces, x=ap, h=12, test_forward=True, level=[90, 80])
#test alias argument
test_eq(
    repr(AutoCES()),
    'CES'
)
test_eq(
    repr(AutoCES(alias='AutoCES_custom')),
    'AutoCES_custom'
)
show_doc(AutoCES, title_level=3)
show_doc(AutoCES.fit, title_level=3)
show_doc(AutoCES.predict, title_level=3)
show_doc(AutoCES.predict_in_sample, title_level=3)
show_doc(AutoCES.forecast, title_level=3)
show_doc(AutoCES.forward, title_level=3)
from statsforecast.models import AutoCES
from statsforecast.utils import AirPassengers as ap

# CES' usage example
# Multiplicative trend, optimal error and seasonality
ces = AutoCES(model='Z',  
              season_length=4)
ces = ces.fit(y=ap)
y_hat_dict = ces.predict(h=4)
y_hat_dict
theta = AutoTheta(season_length=12)
test_class(theta, x=ap, h=12, level=[80, 90], test_forward=True)
fcst_theta = theta.forecast(ap, 13, None, None, (80,95), True)
_plot_insample_pi(fcst_theta)
# test conformal prediction
theta_c = AutoTheta(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(theta_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_theta_c = theta_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_theta_c['mean'],
    fcst_theta['mean'],
)
_plot_fcst(fcst_theta_c)
zero_theta = theta.forward(np.zeros(10), h=12, level=[80, 90], fitted=True)
#test alias argument
test_eq(
    repr(AutoTheta()),
    'AutoTheta'
)
test_eq(
    repr(AutoTheta(alias='AutoTheta_custom')),
    'AutoTheta_custom'
)
show_doc(AutoTheta, title_level=3)
show_doc(AutoTheta.fit, title_level=3)
show_doc(AutoTheta.predict, title_level=3)
show_doc(AutoTheta.predict_in_sample, title_level=3)
show_doc(AutoTheta.forecast, title_level=3)
show_doc(AutoTheta.forward, title_level=3)
from statsforecast.models import AutoTheta
from statsforecast.utils import AirPassengers as ap

# AutoTheta's usage example
theta = AutoTheta(season_length=4)
theta = theta.fit(y=ap)
y_hat_dict = theta.predict(h=4)
y_hat_dict
show_doc(AutoMFLES)
show_doc(AutoMFLES.fit)
show_doc(AutoMFLES.predict)
show_doc(AutoMFLES.predict_in_sample)
show_doc(AutoMFLES.forecast)
deg_ts = np.zeros(10)
h = 12
X = np.random.rand(ap.size, 2)
X_future = np.random.rand(h, 2)

auto_mfles = AutoMFLES(test_size=h, season_length=12)
test_class(auto_mfles, x=deg_ts, X=X, X_future=X_future, h=h, skip_insample=False, test_forward=False)

auto_mfles = AutoMFLES(test_size=h, season_length=12, prediction_intervals=ConformalIntervals(h=h, n_windows=2))
test_class(auto_mfles, x=ap, X=X, X_future=X_future, h=h, skip_insample=False, level=[80, 95], test_forward=False)
fcst_auto_mfles = auto_mfles.forecast(ap, h, X=X, X_future=X_future, fitted=True, level=[80, 95])
_plot_insample_pi(fcst_auto_mfles)
tbats = AutoTBATS(season_length=12)
test_class(tbats, x=ap, h=12, level=[90, 80])
fcst_tbats = tbats.forecast(ap, 13, None, None, (80,95), True)
_plot_fcst(fcst_tbats)
_plot_insample_pi(fcst_tbats)
show_doc(AutoTBATS, title_level=3)
show_doc(AutoTBATS.fit, title_level=3)
show_doc(AutoTBATS.predict, title_level=3)
show_doc(AutoTBATS.predict_in_sample, title_level=3)
show_doc(AutoTBATS.forecast, title_level=3)
simple_arima = ARIMA(order=(1, 0, 0), season_length=12) 
test_class(simple_arima, x=ap, h=12, level=[90, 80], test_forward=True)
fcst_simple_arima = simple_arima.forecast(ap, 13, None, None, (80,95), True)
_plot_insample_pi(fcst_simple_arima)
# test conformal prediction
simple_arima_c = ARIMA(order=(1, 0, 0), season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(simple_arima_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_simple_arima_c = simple_arima_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_simple_arima_c['mean'],
    fcst_simple_arima['mean'],
)
_plot_fcst(fcst_simple_arima_c)
simple_arima = ARIMA(order=(2, 0, 0), season_length=12, fixed={'ar1': 0.5, 'ar2': 0.5}) 
test_class(simple_arima, x=ap, h=12, level=[90, 80], test_forward=True)
fcst_simple_arima = simple_arima.forecast(ap, 4, None, None, (80,95), True)
_plot_insample_pi(fcst_simple_arima)
test_eq(
    fcst_simple_arima['mean'],
    np.array([411., 421.5, 416.25, 418.875])
)
#test alias argument
test_eq(
    repr(ARIMA()),
    'ARIMA'
)
test_eq(
    repr(ARIMA(alias='ARIMA_seasonality')),
    'ARIMA_seasonality'
)
show_doc(ARIMA, title_level=3)
show_doc(ARIMA.fit, title_level=3)
show_doc(ARIMA.predict, title_level=3)
show_doc(ARIMA.predict_in_sample, title_level=3)
show_doc(ARIMA.forecast, title_level=3)
show_doc(ARIMA.forward, title_level=3)
from statsforecast.models import ARIMA
from statsforecast.utils import AirPassengers as ap

# ARIMA's usage example
arima = ARIMA(order=(1, 0, 0), season_length=12)
arima = arima.fit(y=ap)
y_hat_dict = arima.predict(h=4, level=[80])
y_hat_dict
ar = AutoRegressive(lags=[12], fixed={'ar12': 0.9999999}) 
test_class(ar, x=ap, h=12, level=[90, 80], test_forward=True)
fcst_ar = ar.forecast(ap, 13, None, None, (80,95), True)
# we should recover seasonal naive
test_close(
    fcst_ar['mean'][:-1],
    ap[-12:], 
    eps=1e-4
)
_plot_insample_pi(fcst_simple_arima)
# test conformal prediction
ar_c = AutoRegressive(lags=[12], fixed={'ar12': 0.9999999}, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(ar_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_ar_c = ar_c.forecast(ap, 13, None, None, (80,95), True)
#| hide
test_eq(
    fcst_ar_c['mean'],
    fcst_ar['mean'],
)
_plot_fcst(fcst_ar_c)
#test alias argument
test_eq(
    repr(AutoRegressive(lags=[12])),
    'AutoRegressive'
)
test_eq(
    repr(AutoRegressive(lags=[12], alias='AutoRegressive_lag12')),
    'AutoRegressive_lag12'
)
show_doc(AutoRegressive, title_level=3)
show_doc(AutoRegressive.fit, title_level=3, name='AutoRegressive.fit')
show_doc(AutoRegressive.predict, title_level=3, name='AutoRegressive.predict')
show_doc(AutoRegressive.predict_in_sample, title_level=3, name='AutoRegressive.predict_in_sample')
show_doc(AutoRegressive.forecast, title_level=3, name='AutoRegressive.forecast')
show_doc(AutoRegressive.forward, title_level=3, name='AutoRegressive.forward')
from statsforecast.models import AutoRegressive
from statsforecast.utils import AirPassengers as ap

# AutoRegressive's usage example
ar = AutoRegressive(lags=[12])
ar = ar.fit(y=ap)
y_hat_dict = ar.predict(h=4, level=[80])
y_hat_dict
ses = SimpleExponentialSmoothing(alpha=0.1)
test_class(ses, x=ap, h=12)
#more tests
ses = ses.fit(ap)
fcst_ses = ses.predict(12)
test_close(fcst_ses['mean'], np.repeat(460.3028, 12), eps=1e-4)
#to recover these residuals from R
#you have to pass initial="simple"
#in the `ses` function
np.testing.assert_allclose(
    ses.predict_in_sample()['fitted'][[0, 1, -1]], 
    np.array([np.nan, 118 - 6., 432 + 31.447525])
)
# test conformal prediction
ses_c = SimpleExponentialSmoothing(alpha=0.1, prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(ses_c, x=ap, h=13, level=[90, 80], test_forward=False, skip_insample=True)
fcst_ses_c = ses_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_ses_c['mean'][:12],
    fcst_ses['mean']
)
_plot_fcst(fcst_ses_c)
#test alias argument
test_eq(
    repr(SimpleExponentialSmoothing(alpha=0.1)),
    'SES'
)
test_eq(
    repr(SimpleExponentialSmoothing(alpha=0.1, alias='SES_custom')),
    'SES_custom'
)
show_doc(SimpleExponentialSmoothing, title_level=3)
show_doc(SimpleExponentialSmoothing.forecast, title_level=3)
show_doc(SimpleExponentialSmoothing.fit, title_level=3)
show_doc(SimpleExponentialSmoothing.predict, title_level=3)
show_doc(SimpleExponentialSmoothing.predict_in_sample, title_level=3)
from statsforecast.models import SimpleExponentialSmoothing
from statsforecast.utils import AirPassengers as ap

# SimpleExponentialSmoothing's usage example
ses = SimpleExponentialSmoothing(alpha=0.5)
ses = ses.fit(y=ap)
y_hat_dict = ses.predict(h=4)
y_hat_dict
ses_op = SimpleExponentialSmoothingOptimized()
test_class(ses_op, x=ap, h=12)
ses_op = ses_op.fit(ap)
fcst_ses_op = ses_op.predict(12)
# test conformal prediction
ses_op_c = SimpleExponentialSmoothingOptimized(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(ses_op_c, x=ap, h=13, level=[90, 80], skip_insample=True)
fcst_ses_op_c = ses_op_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_ses_op_c['mean'][:12],
    fcst_ses_op['mean']
    )
_plot_fcst(fcst_ses_op_c)
#test alias argument
test_eq(
    repr(SimpleExponentialSmoothingOptimized()),
    'SESOpt'
)
test_eq(
    repr(SimpleExponentialSmoothingOptimized(alias='SESOpt_custom')),
    'SESOpt_custom'
)
show_doc(SimpleExponentialSmoothingOptimized, title_level=3)
show_doc(SimpleExponentialSmoothingOptimized.fit, title_level=3)
show_doc(SimpleExponentialSmoothingOptimized.predict, title_level=3)
show_doc(SimpleExponentialSmoothingOptimized.predict_in_sample, title_level=3)
show_doc(SimpleExponentialSmoothingOptimized.forecast, title_level=3)
from statsforecast.models import SimpleExponentialSmoothingOptimized
from statsforecast.utils import AirPassengers as ap

# SimpleExponentialSmoothingOptimized's usage example
seso = SimpleExponentialSmoothingOptimized()
seso = seso.fit(y=ap)
y_hat_dict = seso.predict(h=4)
y_hat_dict
seas_es = SeasonalExponentialSmoothing(season_length=12, alpha=1.)
test_class(seas_es, x=ap, h=12)
test_eq(seas_es.predict_in_sample()['fitted'][-3:],  np.array([461 - 54., 390 - 28., 432 - 27.]))
seas_es = seas_es.fit(ap)
fcst_seas_es = seas_es.predict(12)
# test conformal prediction
seas_es_c = SeasonalExponentialSmoothing(season_length=12, alpha=1., prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(seas_es_c, x=ap, h=13, level=[90, 80], test_forward=False, skip_insample=True)
fcst_seas_es_c = seas_es_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_seas_es_c['mean'][:12],
    fcst_seas_es['mean']
)
_plot_fcst(fcst_seas_es_c)
# test we can recover the expected seasonality
test_eq(
    seas_es.forecast(ap[4:], h=12)['mean'],
    seas_es.forecast(ap, h=12)['mean']
)
# test close to seasonal naive
for i in range(1, 13):
    test_close(
        ap[i:][-12:],
        seas_es.forecast(ap[i:], h=12)['mean'],
    )
plt.plot(np.concatenate([ap[6:], seas_es.forecast(ap[6:], h=12)['mean']]))
#test alias argument
test_eq(
    repr(SeasonalExponentialSmoothing(season_length=12, alpha=1.)),
    'SeasonalES'
)
test_eq(
    repr(SeasonalExponentialSmoothing(season_length=12, alpha=1., alias='SeasonalES_custom')),
    'SeasonalES_custom'
)
show_doc(SeasonalExponentialSmoothing, title_level=3)
show_doc(SeasonalExponentialSmoothing.fit, title_level=3)
show_doc(SeasonalExponentialSmoothing.predict, title_level=3)
show_doc(SeasonalExponentialSmoothing.predict_in_sample, title_level=3)
show_doc(SeasonalExponentialSmoothing.forecast, title_level=3)
from statsforecast.models import SeasonalExponentialSmoothing
from statsforecast.utils import AirPassengers as ap

# SeasonalExponentialSmoothing's usage example
model = SeasonalExponentialSmoothing(alpha=0.5, season_length=12)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
seas_es_opt = SeasonalExponentialSmoothingOptimized(season_length=12)
test_class(seas_es_opt, x=ap, h=12)
fcst_seas_es_opt = seas_es_opt.forecast(ap, h=12)
# test conformal prediction
seas_es_opt_c = SeasonalExponentialSmoothingOptimized(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(seas_es_opt_c, x=ap, h=13, level=[90, 80], test_forward=False, skip_insample=True)
fcst_seas_es_opt_c = seas_es_opt_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_seas_es_opt_c['mean'][:12],
    fcst_seas_es_opt['mean']
)
_plot_fcst(fcst_seas_es_opt_c)
for i in range(1, 13):
    test_close(
        ap[i:][-12:],
        seas_es_opt.forecast(ap[i:], h=12)['mean'],
        eps=0.8
    )
#test alias argument
test_eq(
    repr(SeasonalExponentialSmoothingOptimized(season_length=12)),
    'SeasESOpt'
)
test_eq(
    repr(SeasonalExponentialSmoothingOptimized(season_length=12, alias='SeasESOpt_custom')),
    'SeasESOpt_custom'
)
show_doc(SeasonalExponentialSmoothingOptimized, title_level=3)
show_doc(SeasonalExponentialSmoothingOptimized.forecast, title_level=3)
show_doc(SeasonalExponentialSmoothingOptimized.fit, title_level=3)
show_doc(SeasonalExponentialSmoothingOptimized.predict, title_level=3)
show_doc(SeasonalExponentialSmoothingOptimized.predict_in_sample, title_level=3)
from statsforecast.models import SeasonalExponentialSmoothingOptimized
from statsforecast.utils import AirPassengers as ap

# SeasonalExponentialSmoothingOptimized's usage example
model = SeasonalExponentialSmoothingOptimized(season_length=12)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
holt = Holt(season_length=12, error_type='A')
fcast_holt = holt.forecast(ap,12)

ets = AutoETS(season_length=12, model='AAN')
fcast_ets = ets.forecast(ap,12)

np.testing.assert_equal(
    fcast_holt, 
    fcast_ets
)
holt = Holt(season_length=12, error_type='A')
holt.fit(ap)
fcast_holt = holt.predict(12)

ets = AutoETS(season_length=12, model='AAN')
fcast_ets = ets.forecast(ap,12)

np.testing.assert_equal(
    fcast_holt, 
    fcast_ets
)
holt_c = Holt(season_length=12, error_type='A', prediction_intervals=ConformalIntervals(h=12, n_windows=2))
fcast_holt_c = holt_c.forecast(ap, 12, level=[80, 90])

ets_c = AutoETS(season_length=12, model='AAN', prediction_intervals=ConformalIntervals(h=12, n_windows=2))
fcast_ets_c = ets_c.forecast(ap, 12, level=[80, 90])

np.testing.assert_equal(
    fcast_holt_c, 
    fcast_ets_c,
)
#test alias argument
test_eq(
    repr(Holt()),
    'Holt'
)
test_eq(
    repr(Holt(alias='Holt_custom')),
    'Holt_custom'
)
show_doc(Holt, title_level=3)
show_doc(Holt.forecast, name='Holt.forecast', title_level=3)
show_doc(Holt.fit, name='Holt.fit', title_level=3)
show_doc(Holt.predict, name='Holt.predict', title_level=3)
show_doc(Holt.predict_in_sample, name='Holt.predict_in_sample', title_level=3)
show_doc(Holt.forward, name='Holt.forward', title_level=3)
from statsforecast.models import Holt
from statsforecast.utils import AirPassengers as ap

# Holt's usage example
model = Holt(season_length=12, error_type='A')
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
hw = HoltWinters(season_length=12, error_type='A')
fcast_hw = hw.forecast(ap,12)

ets = AutoETS(season_length=12, model='AAA')
fcast_ets = ets.forecast(ap,12)

np.testing.assert_equal(
    fcast_hw, 
    fcast_ets
)
hw_c = HoltWinters(season_length=12, error_type='A', prediction_intervals=ConformalIntervals(h=12, n_windows=2))
fcast_hw_c = hw_c.forecast(ap, 12, level=[80, 90])

ets_c = AutoETS(season_length=12, model='AAA', prediction_intervals=ConformalIntervals(h=12, n_windows=2))
fcast_ets_c = ets_c.forecast(ap, 12, level=[80, 90])

np.testing.assert_equal(
    fcast_hw_c, 
    fcast_ets_c,
)
hw = HoltWinters(season_length=12, error_type='A')
hw.fit(ap)
fcast_hw = hw.predict(12)

ets = AutoETS(season_length=12, model='AAA')
fcast_ets = ets.forecast(ap,12)

np.testing.assert_equal(
    fcast_hw, 
    fcast_ets
)
#test alias argument
test_eq(
    repr(HoltWinters()),
    'HoltWinters'
)
test_eq(
    repr(HoltWinters(alias='HoltWinters_custom')),
    'HoltWinters_custom'
)
show_doc(HoltWinters, title_level=3)
show_doc(HoltWinters.forecast, name='HoltWinters.forecast', title_level=3)
show_doc(HoltWinters.fit, name='HoltWinters.fit', title_level=3) 
show_doc(HoltWinters.predict, name='HoltWinters.predict', title_level=3)
show_doc(HoltWinters.predict_in_sample, name= 'HoltWinters.predict_in_sample', title_level=3)
show_doc(HoltWinters.forward, name='HoltWinters.forward', title_level=3)
from statsforecast.models import HoltWinters
from statsforecast.utils import AirPassengers as ap

# Holt-Winters' usage example
model = HoltWinters(season_length=12, error_type='A')
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
ha = HistoricAverage()
test_class(ha, x=ap, h=12, level=[80, 90])
#more tests
ha.fit(ap)
fcst_ha = ha.predict(12)
test_close(fcst_ha['mean'], np.repeat(ap.mean(), 12), eps=1e-5)
np.testing.assert_almost_equal(
    ha.predict_in_sample()['fitted'][:4],
    #np.array([np.nan, 112., 115., 120.6666667]), 
    np.array([280.2986,280.2986,280.2986,280.2986]), 
    decimal=4
)
ha = HistoricAverage()
fcst_ha = ha.forecast(ap,12,None,None,(80,95), True)
np.testing.assert_almost_equal(
    fcst_ha['lo-80'],
    np.repeat(126.0227,12),
    decimal=4
)
_plot_insample_pi(fcst_ha)
# test conformal prediction
ha_c = HistoricAverage(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(ha_c, x=ap, h=13, level=[90, 80], test_forward=False)
fcst_ha_c = ha_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_ha_c['mean'][:12],
    fcst_ha['mean'],
)
_plot_fcst(fcst_ha_c)
#test alias argument
test_eq(
    repr(HistoricAverage()),
    'HistoricAverage'
)
test_eq(
    repr(HistoricAverage(alias='HistoricAverage_custom')),
    'HistoricAverage_custom'
)
show_doc(HistoricAverage, title_level=3)
show_doc(HistoricAverage.forecast, title_level=3)
show_doc(HistoricAverage.fit, title_level=3)
show_doc(HistoricAverage.predict, title_level=3)
show_doc(HistoricAverage.predict_in_sample, title_level=3)
from statsforecast.models import HistoricAverage
from statsforecast.utils import AirPassengers as ap

# HistoricAverage's usage example
model = HistoricAverage()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
# Test prediction intervals - forecast
naive = Naive()
naive.forecast(ap, 12)
naive.forecast(ap, 12, None, None, (80,95), True)
# Test prediction intervals - fit & predict
naive.fit(ap)
naive.predict(12)
naive.predict(12, None, (80,95))
naive = Naive()
test_class(naive, x=ap, h=12, level=[90, 80])
naive.fit(ap)
fcst_naive = naive.predict(12)
test_close(fcst_naive['mean'], np.repeat(ap[-1], 12), eps=1e-5)
naive = Naive()
fcst_naive = naive.forecast(ap,12,None,None,(80,95), True)
np.testing.assert_almost_equal(
    fcst_naive['lo-80'],
    np.array([388.7984, 370.9037, 357.1726, 345.5967, 335.3982, 326.1781, 317.6992, 309.8073, 302.3951, 295.3845, 288.7164, 282.3452]),
    decimal=4
) # this is almost equal since Hyndman's forecasts are rounded up to 4 decimals
_plot_insample_pi(fcst_naive)
# Unit test forward:=forecast
naive = Naive()
fcst_naive = naive.forward(ap,12,None,None,(80,95), True)
np.testing.assert_almost_equal(
    fcst_naive['lo-80'],
    np.array([388.7984, 370.9037, 357.1726, 345.5967, 335.3982, 326.1781, 317.6992, 309.8073, 302.3951, 295.3845, 288.7164, 282.3452]),
    decimal=4
) # this is almost equal since Hyndman's forecasts are rounded up to 4 decimals
# test conformal prediction
naive_c = Naive(prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(naive_c, x=ap, h=13, level=[90, 80], test_forward=False)
fcst_naive_c = naive_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_naive_c['mean'][:12],
    fcst_naive['mean']
)
_plot_fcst(fcst_naive_c)
#test alias argument
test_eq(
    repr(Naive()),
    'Naive'
)
test_eq(
    repr(Naive(alias='Naive_custom')),
    'Naive_custom'
)
show_doc(Naive, title_level=3)
show_doc(Naive.forecast, title_level=3)
show_doc(Naive.fit, title_level=3)
show_doc(Naive.predict, title_level=3)
show_doc(Naive.predict_in_sample, title_level=3)
from statsforecast.models import Naive
from statsforecast.utils import AirPassengers as ap

# Naive's usage example
model = Naive()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
# Test prediction intervals - forecast
rwd = RandomWalkWithDrift()
rwd.forecast(ap, 12)
rwd.forecast(ap, 12, None, None, (80,95), True)
# Test prediction intervals - fit & predict 
rwd = RandomWalkWithDrift()
rwd.fit(ap)
rwd.predict(12)
rwd.predict(12, None, (80,95))
rwd = RandomWalkWithDrift()
test_class(rwd, x=ap, h=12, level=[90, 80])
rwd = rwd.fit(ap)
fcst_rwd = rwd.predict(12)
test_close(fcst_rwd['mean'][:2], np.array([434.2378, 436.4755]), eps=1e-4)
np.testing.assert_almost_equal(
    rwd.predict_in_sample()['fitted'][:3], 
    np.array([np.nan, 118 - 3.7622378, 132 - 11.7622378]),
    decimal=6
)
# test conformal prediction
rwd_c = RandomWalkWithDrift(prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(rwd_c, x=ap, h=13, level=[90, 80], test_forward=False)
fcst_rwd_c = rwd_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_rwd_c['mean'][:12],
    fcst_rwd['mean']
)
_plot_fcst(fcst_rwd_c)
rwd = RandomWalkWithDrift()
fcst_rwd = rwd.forecast(y=ap, h=12, X=None, X_future=None, level=(80,95), fitted=True)
np.testing.assert_almost_equal(
    fcst_rwd['lo-80'],
    np.array([390.9799, 375.0862, 363.2664, 353.5325, 345.1178, 337.6304, 330.8384, 324.5916, 318.7857, 313.3453, 308.2136, 303.3469]),
    decimal=1
)
_plot_insample_pi(fcst_rwd)
#test alias argument
test_eq(
    repr(RandomWalkWithDrift()),
    'RWD'
)
test_eq(
    repr(RandomWalkWithDrift(alias='RWD_custom')),
    'RWD_custom'
)
show_doc(RandomWalkWithDrift, title_level=3)
show_doc(RandomWalkWithDrift.forecast, title_level=3)
show_doc(RandomWalkWithDrift.fit, title_level=3)
show_doc(RandomWalkWithDrift.predict, title_level=3)
show_doc(RandomWalkWithDrift.predict_in_sample, title_level=3)
from statsforecast.models import RandomWalkWithDrift
from statsforecast.utils import AirPassengers as ap

# RandomWalkWithDrift's usage example
model = RandomWalkWithDrift()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
# Test prediction intervals - forecast
seas_naive = SeasonalNaive(season_length=12)
seas_naive.forecast(ap, 12)
seas_naive.forecast(ap, 12, None, None, (80,95), True)
# Test prediction intervals - fit and predict 
seas_naive = SeasonalNaive(season_length=12)
seas_naive.fit(ap)
seas_naive.predict(12)
seas_naive.predict(12, None, (80,95))
seas_naive = SeasonalNaive(season_length=12)
test_class(seas_naive, x=ap, h=12, level=[90, 80])
seas_naive = seas_naive.fit(ap)
fcst_seas_naive = seas_naive.predict(12)
test_eq(seas_naive.predict_in_sample()['fitted'][-3:], np.array([461 - 54., 390 - 28., 432 - 27.]))
seas_naive = SeasonalNaive(season_length=12)
fcst_seas_naive = seas_naive.forecast(y=ap, h=12, X=None, X_future=None, level=(80,95), fitted=True)
np.testing.assert_almost_equal(
    fcst_seas_naive['lo-80'],
    np.array([370.4595, 344.4595, 372.4595, 414.4595, 425.4595, 488.4595, 
              575.4595, 559.4595, 461.4595, 414.4595, 343.4595, 385.4595]),
    decimal=4
)
_plot_insample_pi(fcst_seas_naive)
# Unit test forward:=forecast
seas_naive = SeasonalNaive(season_length=12)
fcst_seas_naive = seas_naive.forward(y=ap, h=12, X=None, X_future=None, level=(80,95), fitted=True)
np.testing.assert_almost_equal(
    fcst_seas_naive['lo-80'],
    np.array([370.4595, 344.4595, 372.4595, 414.4595, 425.4595, 488.4595, 
              575.4595, 559.4595, 461.4595, 414.4595, 343.4595, 385.4595]),
    decimal=4
) # this is almost equal since Hyndman's forecasts are rounded up to 4 decimals
# test h > season_length
seas_naive_bigh = SeasonalNaive(season_length=12)
fcst_seas_naive_bigh = seas_naive_bigh.forecast(y=ap, h=13, X=None, X_future=None, level=(80,95), fitted=True)
np.testing.assert_almost_equal(
    fcst_seas_naive_bigh['lo-80'][:12],
    np.array([370.4595, 344.4595, 372.4595, 414.4595, 425.4595, 488.4595, 
              575.4595, 559.4595, 461.4595, 414.4595, 343.4595, 385.4595]),
    decimal=4
)
_plot_fcst(fcst_seas_naive_bigh)
# test conformal prediction
seas_naive_c = SeasonalNaive(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(seas_naive_c, x=ap, h=13, level=[90, 80], test_forward=False)
fcst_seas_naive_c = seas_naive_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_seas_naive_c['mean'][:12],
    fcst_seas_naive['mean']
)
_plot_fcst(fcst_seas_naive_c)
#test alias argument
test_eq(
    repr(SeasonalNaive(12)),
    'SeasonalNaive'
)
test_eq(
    repr(SeasonalNaive(12, alias='SeasonalNaive_custom')),
    'SeasonalNaive_custom'
)
show_doc(SeasonalNaive, title_level=3)
show_doc(SeasonalNaive.forecast, title_level=3)
show_doc(SeasonalNaive.fit, title_level=3)
show_doc(SeasonalNaive.predict, title_level=3)
show_doc(SeasonalNaive.predict_in_sample, title_level=3)
from statsforecast.models import SeasonalNaive
from statsforecast.utils import AirPassengers as ap

# SeasonalNaive's usage example
model = SeasonalNaive(season_length=12)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
w_avg = WindowAverage(window_size=24)
test_class(w_avg, x=ap, h=12, skip_insample=True)
w_avg = w_avg.fit(ap)
fcst_w_avg = w_avg.predict(12)
test_close(fcst_w_avg['mean'], np.repeat(ap[-24:].mean(), 12))
# test conformal prediction
w_avg_c = WindowAverage(window_size=24, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(w_avg_c, x=ap, h=13, level=[90, 80], test_forward=False, skip_insample=True)
fcst_w_avg_c = w_avg_c.forecast(ap, 13, None, None, (80,95), False)
test_eq(
    fcst_w_avg_c['mean'][:12],
    fcst_w_avg['mean']
)
_plot_fcst(fcst_w_avg_c)
#test alias argument
test_eq(
    repr(WindowAverage(1)),
    'WindowAverage'
)
test_eq(
    repr(WindowAverage(1, alias='WindowAverage_custom')),
    'WindowAverage_custom'
)
show_doc(WindowAverage, title_level=3)
show_doc(WindowAverage.forecast, title_level=3)
show_doc(WindowAverage.fit, title_level=3)
show_doc(WindowAverage.predict, title_level=3)
from statsforecast.models import WindowAverage
from statsforecast.utils import AirPassengers as ap

# WindowAverage's usage example
model = WindowAverage(window_size=12*4)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
seas_w_avg = SeasonalWindowAverage(season_length=12, window_size=1)
test_class(seas_w_avg, x=ap, h=12, skip_insample=True)
seas_w_avg = seas_w_avg.fit(ap)
fcst_seas_w_avg = seas_w_avg.predict(12)
# test conformal prediction
seas_w_avg_c = SeasonalWindowAverage(season_length=12, window_size=1, prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(seas_w_avg_c, x=ap, h=13, level=[90, 80], test_forward=False, skip_insample=True)
fcst_seas_w_avg_c = seas_w_avg_c.forecast(ap, 13, None, None, (80,95), False)
test_eq(
    fcst_seas_w_avg_c['mean'][:12],
    fcst_seas_w_avg['mean']
)
fcst_seas_w_avg_c['mean'][:12]
_plot_fcst(fcst_seas_w_avg_c)
#test alias argument
test_eq(
    repr(SeasonalWindowAverage(12, 1)),
    'SeasWA'
)
test_eq(
    repr(SeasonalWindowAverage(12, 1, alias='SeasWA_custom')),
    'SeasWA_custom'
)
show_doc(SeasonalWindowAverage, title_level=3)
show_doc(SeasonalWindowAverage.forecast, title_level=3)
show_doc(SeasonalWindowAverage.fit, title_level=3)
show_doc(SeasonalWindowAverage.predict, title_level=3)
from statsforecast.models import SeasonalWindowAverage
from statsforecast.utils import AirPassengers as ap

# SeasonalWindowAverage's usage example
model = SeasonalWindowAverage(season_length=12, window_size=4)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
adida = ADIDA()
test_class(adida, x=ap, h=12, skip_insample=False)
test_class(adida, x=deg_ts, h=12, skip_insample=False)
fcst_adida = adida.forecast(ap, 12)

_test_fitted_sparse(ADIDA)
# test conformal prediction
adida_c = ADIDA(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(adida_c, x=ap, h=13, level=[90, 80], skip_insample=False)
fcst_adida_c = adida_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_adida_c['mean'][:12],
    fcst_adida['mean'],
)
_plot_insample_pi(fcst_adida_c)
_plot_fcst(fcst_adida_c)
#test alias argument
test_eq(
    repr(ADIDA()),
    'ADIDA'
)
test_eq(
    repr(ADIDA(alias='ADIDA_custom')),
    'ADIDA_custom'
)
show_doc(ADIDA, title_level=3)
show_doc(ADIDA.forecast, title_level=3)
show_doc(ADIDA.fit, title_level=3)
show_doc(ADIDA.predict, title_level=3)
from statsforecast.models import ADIDA
from statsforecast.utils import AirPassengers as ap

# ADIDA's usage example
model = ADIDA()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
croston = CrostonClassic(prediction_intervals=ConformalIntervals(2, 1))
test_class(croston, x=ap, h=12, skip_insample=False, level=[80])
test_class(croston, x=deg_ts, h=12, skip_insample=False, level=[80])
fcst_croston = croston.forecast(ap, 12)

_test_fitted_sparse(CrostonClassic)
# test conformal prediction
croston_c = CrostonClassic(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(croston_c, x=ap, h=13, level=[90.0, 80.0], skip_insample=False)
fcst_croston_c = croston_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_croston_c['mean'][:12],
    fcst_croston['mean'],
)
_plot_insample_pi(fcst_croston_c)
_plot_fcst(fcst_croston_c)
#test alias argument
test_eq(
    repr(CrostonClassic()),
    'CrostonClassic'
)
test_eq(
    repr(CrostonClassic(alias='CrostonClassic_custom')),
    'CrostonClassic_custom'
)
show_doc(CrostonClassic, title_level=3)
show_doc(CrostonClassic.forecast, title_level=3)
show_doc(CrostonClassic.fit, title_level=3)
show_doc(CrostonClassic.predict, title_level=3)
from statsforecast.models import CrostonClassic
from statsforecast.utils import AirPassengers as ap

# CrostonClassic's usage example
model = CrostonClassic()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
croston_op = CrostonOptimized(prediction_intervals=ConformalIntervals(2, 1))
test_class(croston_op, x=ap, h=12, skip_insample=False, level=[80])
test_class(croston_op, x=deg_ts, h=12, skip_insample=False, level=[80])
fcst_croston_op = croston_op.forecast(ap, 12)

_test_fitted_sparse(CrostonOptimized)
# test conformal prediction
croston_op_c = CrostonOptimized(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(croston_op_c, x=ap, h=13, level=[90, 80], skip_insample=False)
fcst_croston_op_c = croston_op_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_croston_op_c['mean'][:12],
    fcst_croston_op['mean'],
)
_plot_insample_pi(fcst_croston_op_c)
_plot_fcst(fcst_croston_op_c)
#test alias argument
test_eq(
    repr(CrostonOptimized()),
    'CrostonOptimized'
)
test_eq(
    repr(CrostonOptimized(alias='CrostonOptimized_custom')),
    'CrostonOptimized_custom'
)
show_doc(CrostonOptimized, title_level=3)
show_doc(CrostonOptimized.forecast, title_level=3)
show_doc(CrostonOptimized.fit, title_level=3)
show_doc(CrostonOptimized.predict, title_level=3)
from statsforecast.models import CrostonOptimized
from statsforecast.utils import AirPassengers as ap

# CrostonOptimized's usage example
model = CrostonOptimized()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
croston_sba = CrostonSBA(prediction_intervals=ConformalIntervals(2, 1))
test_class(croston_sba, x=ap, h=12, skip_insample=False, level=[80])
test_class(croston_sba, x=deg_ts, h=12, skip_insample=False, level=[80])
fcst_croston_sba = croston_sba.forecast(ap, 12)

_test_fitted_sparse(CrostonSBA)
# test conformal prediction
croston_sba_c = CrostonSBA(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(croston_sba_c, x=ap, h=13, level=[90, 80], skip_insample=False)
fcst_croston_sba_c = croston_sba_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_croston_sba_c['mean'][:12],
    fcst_croston_sba['mean'],
)
_plot_insample_pi(fcst_croston_sba_c)
_plot_fcst(fcst_croston_sba_c)
#test alias argument
test_eq(
    repr(CrostonSBA()),
    'CrostonSBA'
)
test_eq(
    repr(CrostonSBA(alias='CrostonSBA_custom')),
    'CrostonSBA_custom'
)
show_doc(CrostonSBA, title_level=3)
show_doc(CrostonSBA.forecast, title_level=3)
show_doc(CrostonSBA.fit, title_level=3)
show_doc(CrostonSBA.predict, title_level=3)
from statsforecast.models import CrostonSBA
from statsforecast.utils import AirPassengers as ap

# CrostonSBA's usage example
model = CrostonSBA()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
imapa = IMAPA(prediction_intervals=ConformalIntervals(2, 1))
test_class(imapa, x=ap, h=12, skip_insample=False, level=[80])
test_class(imapa, x=deg_ts, h=12, skip_insample=False, level=[80])
fcst_imapa = imapa.forecast(ap, 12)

_test_fitted_sparse(IMAPA)
# test conformal prediction
imapa_c = IMAPA(prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(imapa_c, x=ap, h=13, level=[90, 80], skip_insample=False)
fcst_imapa_c = imapa_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_imapa_c['mean'][:12],
    fcst_imapa['mean'],
)
_plot_insample_pi(fcst_imapa_c)
_plot_fcst(fcst_imapa_c)
#test alias argument
test_eq(
    repr(IMAPA()),
    'IMAPA'
)
test_eq(
    repr(IMAPA(alias='IMAPA_custom')),
    'IMAPA_custom'
)
show_doc(IMAPA, title_level=3)
show_doc(IMAPA.forecast, title_level=3)
show_doc(IMAPA.fit, title_level=3)
show_doc(IMAPA.predict, title_level=3)
from statsforecast.models import IMAPA
from statsforecast.utils import AirPassengers as ap
# IMAPA's usage example
model = IMAPA()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
tsb = TSB(alpha_d=0.9, alpha_p=0.1, prediction_intervals=ConformalIntervals(2, 1))
test_class(tsb, x=ap, h=12, skip_insample=False, level=[80])
test_class(tsb, x=deg_ts, h=12, skip_insample=False, level=[80])
fcst_tsb = tsb.forecast(ap, 12)

_test_fitted_sparse(lambda: TSB(alpha_d=0.9, alpha_p=0.1))
# test conformal prediction
tsb_c = TSB(alpha_d=0.9, alpha_p=0.1,prediction_intervals=ConformalIntervals(h=13, n_windows=2))
test_class(tsb_c, x=ap, h=13, level=[90, 80], skip_insample=True)
fcst_tsb_c = tsb_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_tsb_c['mean'][:12],
    fcst_tsb['mean'],
)
_plot_insample_pi(fcst_tsb_c)
_plot_fcst(fcst_tsb_c)
#test alias argument
test_eq(
    repr(TSB(0.9, 0.1)),
    'TSB'
)
test_eq(
    repr(TSB(0.9, 0.1, alias='TSB_custom')),
    'TSB_custom'
)
show_doc(TSB, title_level=3)
show_doc(TSB.forecast, title_level=3)
show_doc(TSB.fit, title_level=3)
show_doc(TSB.predict, title_level=3)
from statsforecast.models import TSB
from statsforecast.utils import AirPassengers as ap
# TSB's usage example
model = TSB(alpha_d=0.5, alpha_p=0.5)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
trend_forecasters = [
    AutoARIMA(), 
    AutoCES(), 
    AutoETS(model='ZZN'),
    Naive(),
    CrostonClassic(),
]
skip_insamples = [False, True, False, False, True]
test_forwards = [False, True, True, False, False]
for trend_forecaster, skip_insample, test_forward in zip(trend_forecasters, skip_insamples, test_forwards):
    for stl_kwargs in [None, dict(trend=25)]:
        mstl_model = MSTL(
            season_length=[12, 14], 
            trend_forecaster=trend_forecaster,
            stl_kwargs=stl_kwargs,
        )
        test_class(mstl_model, x=ap, h=12, 
                   skip_insample=skip_insample,
                   level=None,
                   test_forward=test_forward)
# intervals with & without conformal
# trend fcst supports level, use native levels
mstl_native = MSTL(season_length=12, trend_forecaster=ARIMA(order=(0, 1, 0)))
res_native_fp = pd.DataFrame(mstl_native.fit(y=ap).predict(h=24, level=[80, 95]))
res_native_fc = pd.DataFrame(mstl_native.forecast(y=ap, h=24, level=[80, 95]))
pd.testing.assert_frame_equal(res_native_fp, res_native_fc)

# trend fcst supports level, use conformal
mstl_conformal = MSTL(
    season_length=12,
    trend_forecaster=ARIMA(
        order=(0, 1, 0),
        prediction_intervals=ConformalIntervals(h=24),
    ),
)
res_conformal_fp = pd.DataFrame(mstl_conformal.fit(y=ap).predict(h=24, level=[80, 95]))
res_conformal_fc = pd.DataFrame(mstl_conformal.forecast(y=ap, h=24, level=[80, 95]))
pd.testing.assert_frame_equal(res_conformal_fp, res_conformal_fc)
test_fail(lambda: pd.testing.assert_frame_equal(test_native_fp, test_conformal_fp))

# trend fcst doesn't support level
mstl_bad = MSTL(season_length=12, trend_forecaster=CrostonClassic())
test_fail(lambda: mstl_bad.fit(y=ap).predict(h=24, level=[80, 95]), contains='prediction_intervals')
test_fail(lambda: mstl_bad.forecast(y=ap, h=24, level=[80, 95]), contains='prediction_intervals')
# conformal prediction
# define the prediction interval in the trend_forecaster
trend_forecasters = [
    AutoARIMA(prediction_intervals=ConformalIntervals(h=13, n_windows=2)),
    AutoCES(prediction_intervals=ConformalIntervals(h=13, n_windows=2)),
]
skip_insamples = [False, True]
test_forwards = [False, True]
for trend_forecaster, skip_insample, test_forward in zip(trend_forecasters, skip_insamples, test_forwards):
    for stl_kwargs in [None, dict(trend=25)]:
        mstl_model = MSTL(
            season_length=[12, 14], 
            trend_forecaster=trend_forecaster,
            stl_kwargs=stl_kwargs,
        )
        test_class(mstl_model, x=ap, h=13, 
                   skip_insample=skip_insample,
                   level=[80, 90] if not skip_insample else None,
                   test_forward=test_forward)
        _plot_fcst(mstl_model.forecast(ap, 13, None, None, (80,95), False))
# conformal prediction
# define prediction_interval in MSTL
trend_forecasters = [
    AutoCES()
]
for stl_kwargs in [None, dict(trend=25)]:
    mstl_model = MSTL(
        season_length=[12, 14], 
        trend_forecaster=trend_forecaster,
        stl_kwargs=stl_kwargs,
        prediction_intervals=ConformalIntervals(h=13, n_windows=2)
    )
    test_class(mstl_model, x=ap, h=13, 
                skip_insample=False,
                level=[80, 90] if not skip_insample else None,
                test_forward=True)
    _plot_fcst(mstl_model.forecast(ap, 13, None, None, (80,95), False))
#fail with seasonal trend forecasters
test_fail(
    MSTL,
    contains='should not adjust seasonal',
    args=([3, 12], AutoETS(model='ZZZ'))
)
test_fail(
    MSTL,
    contains='should not adjust seasonal',
    args=([3, 12], AutoARIMA(season_length=12))
)
#test alias argument
test_eq(
    repr(MSTL(season_length=7)),
    'MSTL'
)
test_eq(
    repr(MSTL(season_length=7, alias='MSTL_custom')),
    'MSTL_custom'
)
show_doc(MSTL, title_level=3)
show_doc(MSTL.fit, title_level=3)
show_doc(MSTL.predict, title_level=3)
show_doc(MSTL.predict_in_sample, title_level=3)
show_doc(MSTL.forecast, title_level=3)
show_doc(MSTL.forward, title_level=3)
from statsforecast.models import MSTL
from statsforecast.utils import AirPassengers as ap
# MSTL's usage example
mstl_model = MSTL(season_length=[3, 12], trend_forecaster=AutoARIMA(prediction_intervals=ConformalIntervals(h=4, n_windows=2)))
mstl_model = mstl_model.fit(y=ap)
y_hat_dict = mstl_model.predict(h=4, level=[80])
y_hat_dict
show_doc(MFLES)
show_doc(MFLES.fit)
show_doc(MFLES.predict)
show_doc(MFLES.predict_in_sample)
show_doc(MFLES.forecast)
h = 12
X = np.random.rand(ap.size, 2)
X_future = np.random.rand(h, 2)

mfles = MFLES()
test_class(mfles, x=deg_ts, X=X, X_future=X_future, h=h, skip_insample=False, test_forward=False)

mfles = MFLES(prediction_intervals=ConformalIntervals(h=h, n_windows=2))
test_class(mfles, x=ap, X=X, X_future=X_future, h=h, skip_insample=False, level=[80, 95], test_forward=False)
fcst_mfles = mfles.forecast(ap, h, X=X, X_future=X_future, fitted=True, level=[80, 95])
_plot_insample_pi(fcst_mfles)
tbats = TBATS(season_length=12)
test_class(tbats, x=ap, h=12, level=[90, 80])
show_doc(TBATS, title_level=3)
show_doc(TBATS.fit, name='TBATS.fit', title_level=3)
show_doc(TBATS.predict, name='TBATS.predict', title_level=3)
show_doc(TBATS.predict_in_sample, name='TBATS.predict_in_sample', title_level=3)
show_doc(TBATS.forecast, name='TBATS.forecast', title_level=3)
stm = Theta(season_length=12)
fcast_stm = stm.forecast(ap,12)

theta = AutoTheta(season_length=12, model='STM')
fcast_theta = theta.forecast(ap,12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_stm
)
# test conformal prediction
stm_c = Theta(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=5)) 
test_class(stm_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_stm_c = stm_c.forecast(ap, 13, None, None, (80,95), True)

test_eq(
    fcst_stm_c['mean'][:12],
    fcast_stm['mean'],
)
stm = Theta(season_length=12)
stm.fit(ap)
fcast_stm = stm.predict(12)
forward_stm = stm.forward(y=ap, h=12)

theta = AutoTheta(season_length=12, model='STM')
fcast_theta = theta.forecast(ap,12)
theta.fit(ap)
forward_autotheta = theta.forward(y=ap, h=12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_stm,
)
np.testing.assert_equal(
    forward_stm,
    forward_autotheta
)
#test alias argument
test_eq(
    repr(Theta()),
    'Theta'
)
test_eq(
    repr(Theta(alias='Theta_custom')),
    'Theta_custom'
)
show_doc(Theta, title_level=3)
show_doc(Theta.forecast, name='Theta.forecast', title_level=3)
show_doc(Theta.fit, name='Theta.fit', title_level=3)
show_doc(Theta.predict, name='Theta.predict', title_level=3)
show_doc(Theta.predict_in_sample, name='Theta.predict_in_sample', title_level=3)
show_doc(Theta.forward, name='Theta.forward', title_level=3)
from statsforecast.models import Theta
from statsforecast.utils import AirPassengers as ap
# Theta's usage example
model = Theta(season_length=12)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
otm = OptimizedTheta(season_length=12)
fcast_otm = otm.forecast(ap,12)
otm.fit(ap)
forward_otm = otm.forward(y=ap, h=12)

theta = AutoTheta(season_length=12, model='OTM')
fcast_theta = theta.forecast(ap,12)
theta.fit(ap)
forward_autotheta = theta.forward(y=ap, h=12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_otm
)

np.testing.assert_equal(
    forward_autotheta, 
    forward_otm
)
# test conformal prediction
otm_c = OptimizedTheta(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=5)) 
test_class(otm_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_otm_c = otm_c.forecast(ap, 13, None, None, (80,95), True)

test_eq(
    fcst_otm_c['mean'][:12],
    fcast_otm['mean'],
)
otm = OptimizedTheta(season_length=12)
otm.fit(ap)
fcast_otm = otm.predict(12)

theta = AutoTheta(season_length=12, model='OTM')
fcast_theta = theta.forecast(ap,12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_otm
)
#test alias argument
test_eq(
    repr(OptimizedTheta()),
    'OptimizedTheta'
)
test_eq(
    repr(OptimizedTheta(alias='OptimizedTheta_custom')),
    'OptimizedTheta_custom'
)
show_doc(OptimizedTheta, title_level=3)
show_doc(OptimizedTheta.forecast, name='OptimizedTheta.forecast', title_level=3)
show_doc(OptimizedTheta.fit, name='OptimizedTheta.fit', title_level=3)
show_doc(OptimizedTheta.predict, name='OptimizedTheta.predict', title_level=3)
show_doc(OptimizedTheta.predict_in_sample, name='OptimizedTheta.predict_in_sample', title_level=3)
show_doc(OptimizedTheta.forward, name='OptimizedTheta.forward', title_level=3)
from statsforecast.models import OptimizedTheta
from statsforecast.utils import AirPassengers as ap
# OptimzedThetA's usage example
model = OptimizedTheta(season_length=12)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
dstm = DynamicTheta(season_length=12)
fcast_dstm = dstm.forecast(ap,12)
dstm.fit(ap)
forward_dstm = dstm.forward(y=ap, h=12)

theta = AutoTheta(season_length=12, model='DSTM')
fcast_theta = theta.forecast(ap,12)
theta.fit(ap)
forward_autotheta = theta.forward(y=ap, h=12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_dstm
)

np.testing.assert_equal(
    forward_autotheta, 
    forward_dstm
)
dstm = DynamicTheta(season_length=12)
dstm.fit(ap)
fcast_dstm = dstm.predict(12)

theta = AutoTheta(season_length=12, model='DSTM')
fcast_theta = theta.forecast(ap,12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_dstm
)
# test conformal prediction
dstm_c = DynamicTheta(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=5)) 
test_class(dstm_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_dstm_c = dstm_c.forecast(ap, 13, None, None, (80,95), True)

test_eq(
    fcst_dstm_c['mean'][:12],
    fcast_dstm['mean'],
)
#test alias argument
test_eq(
    repr(DynamicTheta()),
    'DynamicTheta'
)
test_eq(
    repr(DynamicTheta(alias='DynamicTheta_custom')),
    'DynamicTheta_custom'
)
show_doc(DynamicTheta, title_level=3)
show_doc(DynamicTheta.forecast, name='DynamicTheta.forecast', title_level=3)
show_doc(DynamicTheta.fit, name='DynamicTheta.fit', title_level=3)
show_doc(DynamicTheta.predict, name='DynamicTheta.predict', title_level=3)
show_doc(DynamicTheta.predict_in_sample, name='DynamicTheta.predict_in_sample', title_level=3)
show_doc(DynamicTheta.forward, name='DynamicTheta.forward', title_level=3)
from statsforecast.models import DynamicTheta
from statsforecast.utils import AirPassengers as ap
# DynStandardThetaMethod's usage example
model = DynamicTheta(season_length=12)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
dotm = DynamicOptimizedTheta(season_length=12)
fcast_dotm = dotm.forecast(ap,12)
dotm.fit(ap)
forward_dotm = dotm.forward(y=ap, h=12)

theta = AutoTheta(season_length=12, model='DOTM')
fcast_theta = theta.forecast(ap,12)
theta.fit(ap)
forward_autotheta = theta.forward(y=ap, h=12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_dotm
)
np.testing.assert_equal(
    forward_autotheta, 
    forward_dotm
)
dotm = DynamicOptimizedTheta(season_length=12)
dotm.fit(ap)
fcast_dotm = dotm.predict(12)

theta = AutoTheta(season_length=12, model='DOTM')
fcast_theta = theta.forecast(ap,12)

np.testing.assert_equal(
    fcast_theta, 
    fcast_dotm
)
# test conformal prediction
dotm_c = DynamicOptimizedTheta(season_length=12, prediction_intervals=ConformalIntervals(h=13, n_windows=5)) 
test_class(dotm_c, x=ap, h=13, level=[90, 80], test_forward=True)
fcst_dotm_c = dotm_c.forecast(ap, 13, None, None, (80,95), True)

test_eq(
    fcst_dotm_c['mean'][:12],
    fcast_dotm['mean'],
)
#test alias argument
test_eq(
    repr(DynamicOptimizedTheta()),
    'DynamicOptimizedTheta'
)
test_eq(
    repr(DynamicOptimizedTheta(alias='DynamicOptimizedTheta_custom')),
    'DynamicOptimizedTheta_custom'
)
show_doc(DynamicOptimizedTheta, title_level=3)
show_doc(DynamicOptimizedTheta.forecast, name='DynamicOptimizedTheta.forecast', title_level=3)
show_doc(DynamicOptimizedTheta.fit, name='DynamicOptimizedTheta.fit', title_level=3)
show_doc(DynamicOptimizedTheta.predict, name='DynamicOptimizedTheta.predict', title_level=3)
show_doc(DynamicOptimizedTheta.predict_in_sample, name='DynamicOptimizedTheta.predict_in_sample', title_level=3)
show_doc(DynamicOptimizedTheta.forward, name='DynamicOptimizedTheta.forward', title_level=3)
from statsforecast.models import DynamicOptimizedTheta
from statsforecast.utils import AirPassengers as ap
# OptimzedThetaMethod's usage example
model = DynamicOptimizedTheta(season_length=12)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
# Generate GARCH(2,2) data 
n = 1000 
w = 0.5
alpha = np.array([0.1, 0.2])
beta = np.array([0.4, 0.2])

y = generate_garch_data(n, w, alpha, beta) 

plt.figure(figsize=(10,4))
plt.plot(y)
garch = GARCH(2,2)
test_class(garch, x=y, h=12, skip_insample=False, level=[90,80])
fcst_garch = garch.forecast(ap, 12)
# test conformal prediction
garch_c = GARCH(2,2,prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(garch_c, x=ap, h=13, level=[90, 80], test_forward=False)
fcst_garch_c = garch_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_garch_c['mean'][:12],
    fcst_garch['mean']
)
_plot_fcst(fcst_garch_c)
h=100
fcst = garch.forecast(y, h=h, level=[80,95], fitted=True)
fig, ax = plt.subplots(1, 1, figsize = (20,7))
#plt.plot(np.arange(0, len(y)), y) 
plt.plot(np.arange(len(y), len(y) + h), fcst['mean'], label='mean')
plt.plot(np.arange(len(y), len(y) + h), fcst['sigma2'], color = 'c', label='sigma2')
plt.plot(np.arange(len(y), len(y) + h), fcst['lo-95'], color = 'r', label='lo-95')
plt.plot(np.arange(len(y), len(y) + h), fcst['hi-95'], color = 'r', label='hi-95')
plt.plot(np.arange(len(y), len(y) + h), fcst['lo-80'], color = 'g', label='lo-80')
plt.plot(np.arange(len(y), len(y) + h), fcst['hi-80'], color = 'g', label='hi-80')
plt.legend()
fig, ax = plt.subplots(1, 1, figsize = (20,7))
plt.plot(np.arange(0, len(y)), y) 
plt.plot(np.arange(0, len(y)), fcst['fitted'], label='fitted') 
plt.plot(np.arange(0, len(y)), fcst['fitted-lo-95'], color = 'r', label='fitted-lo-95')
plt.plot(np.arange(0, len(y)), fcst['fitted-hi-95'], color = 'r', label='fitted-hi-95')
plt.plot(np.arange(0, len(y)), fcst['fitted-lo-80'], color = 'g', label='fitted-lo-80')
plt.plot(np.arange(0, len(y)), fcst['fitted-hi-80'], color = 'g', label='fitted-hi-80')
plt.xlim(len(y)-50, len(y))
plt.legend()
show_doc(GARCH, title_level=3)
show_doc(GARCH.fit, title_level=3)
show_doc(GARCH.predict, title_level=3)
show_doc(GARCH.predict_in_sample, title_level=3)
show_doc(GARCH.forecast, title_level=3)
arch = ARCH(1)
test_class(arch, x=y, h=12, skip_insample=False, level=[90,80])
fcst_arch = arch.forecast(ap, 12)
# test conformal prediction
arch_c = ARCH(1,prediction_intervals=ConformalIntervals(h=13, n_windows=2)) 
test_class(arch_c, x=ap, h=13, level=[90, 80], test_forward=False)
fcst_arch_c = arch_c.forecast(ap, 13, None, None, (80,95), True)
test_eq(
    fcst_arch_c['mean'][:12],
    fcst_arch['mean']
)
_plot_fcst(fcst_arch_c)
garch = GARCH(p=1, q=0)
fcast_garch = garch.forecast(y, h=12, level=[90,80], fitted=True)

arch = ARCH(p=1)
fcast_arch = arch.forecast(y, h=12, level=[90,80], fitted=True)

np.testing.assert_equal(
    fcast_garch, 
    fcast_arch
)
show_doc(ARCH, title_level=3)
show_doc(ARCH.fit, name='ARCH.fit', title_level=3)
show_doc(ARCH.predict, name='ARCH.predict', title_level=3)
show_doc(ARCH.predict_in_sample, name='ARCH.predict_in_sample', title_level=3)
show_doc(ARCH.forecast, name='ARCH.forecast', title_level=3)
from sklearn.linear_model import Ridge
h = 12
skm = SklearnModel(Ridge(), prediction_intervals=ConformalIntervals(h=h))
X = np.arange(ap.size).reshape(-1, 1)
X_future = ap.size + np.arange(h).reshape(-1, 1)
test_class(skm, x=ap, X=X, X_future=X_future, h=h, skip_insample=False, level=[80, 95], test_forward=True)
fcst_skm = skm.forecast(ap, h, X=X, X_future=X_future, fitted=True, level=[80, 95])
_plot_insample_pi(fcst_skm)
#test alias argument
test_eq(
    repr(SklearnModel(Ridge())),
    'Ridge'
)
test_eq(
    repr(SklearnModel(Ridge(), alias='my_ridge')),
    'my_ridge'
)
show_doc(SklearnModel)
show_doc(SklearnModel.fit)
show_doc(SklearnModel.predict)
show_doc(SklearnModel.predict_in_sample)
show_doc(SklearnModel.forecast)
constant_model = ConstantModel(constant=1)
test_class(constant_model, x=ap, h=12, level=[90, 80])
constant_model.forecast(ap, 12, level=[90, 80])
constant_model.forward(ap, 12, level=[90, 80])
#test alias argument
test_eq(
    repr(ConstantModel(1)),
    'ConstantModel'
)
test_eq(
    repr(ConstantModel(1, alias='ConstantModel_custom')),
    'ConstantModel_custom'
)
show_doc(ConstantModel, title_level=3)
show_doc(ConstantModel.forecast, title_level=3)
show_doc(ConstantModel.fit, title_level=3)
show_doc(ConstantModel.predict, title_level=3)
show_doc(ConstantModel.predict_in_sample, title_level=3)
show_doc(ConstantModel.forward, title_level=3)
from statsforecast.models import ConstantModel
from statsforecast.utils import AirPassengers as ap
# ConstantModel's usage example
model = ConstantModel(1)
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
zero_model = ZeroModel()
test_class(constant_model, x=ap, h=12, level=[90, 80])
zero_model.forecast(ap, 12, level=[90, 80])
zero_model.forward(ap, 12, level=[90, 80])
#test alias argument
test_eq(
    repr(ZeroModel()),
    'ZeroModel'
)
test_eq(
    repr(ZeroModel(alias='ZeroModel_custom')),
    'ZeroModel_custom'
)
show_doc(ZeroModel, title_level=3)
show_doc(ZeroModel.forecast, title_level=3, name='ZeroModel.forecast')
show_doc(ZeroModel.fit, title_level=3, name='ZeroModel.fit')
show_doc(ZeroModel.predict, title_level=3, name='ZeroModel.predict')
show_doc(ZeroModel.predict_in_sample, title_level=3, name='ZeroModel.predict_in_sample')
show_doc(ZeroModel.forward, title_level=3, name='ZeroModel.forward')
from statsforecast.models import ZeroModel
from statsforecast.utils import AirPassengers as ap
# NanModel's usage example
model = ZeroModel()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
nanmodel = NaNModel()
nanmodel.forecast(ap, 12, level=[90, 80])
#test alias argument
test_eq(
    repr(NaNModel()),
    'NaNModel'
)
test_eq(
    repr(NaNModel(alias='NaN_custom')),
    'NaN_custom'
)
show_doc(NaNModel, title_level=3)
show_doc(NaNModel.forecast, title_level=3, name='NaNModel.forecast')
show_doc(NaNModel.fit, title_level=3, name='NaNModel.fit')
show_doc(NaNModel.predict, title_level=3, name='NaNModel.predict')
show_doc(NaNModel.predict_in_sample, title_level=3, name='NaNModel.predict_in_sample')
from statsforecast.models import NaNModel
from statsforecast.utils import AirPassengers as ap
# NanModel's usage example
model = NaNModel()
model = model.fit(y=ap)
y_hat_dict = model.predict(h=4)
y_hat_dict
