from statsforecast.utils import AirPassengers as ap
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2
_theta.init_state(ap, _theta.ModelType.STM, initial_smoothed, alpha, theta)
_theta.init_state(ap, _theta.ModelType.OTM, initial_smoothed, alpha, theta)
_theta.init_state(ap, _theta.ModelType.DSTM, initial_smoothed, alpha, theta)
_theta.init_state(ap, _theta.ModelType.DOTM, initial_smoothed, alpha, theta)
#simple theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5))
mse = _theta.calc(
    ap,
    init_states, 
    _theta.ModelType.STM, 
    initial_smoothed,
    alpha,
    theta,
    e_,
    amse_,
    3,
)
#verify we recover the fitted values
np.testing.assert_array_equal(
    ap - e_,
    init_states[:, -1]
)

#verify we get same fitted values than R
# use stm(AirPassengers, s=F, estimation=F, h = 12)
# to recover
np.testing.assert_array_almost_equal(
    init_states[:, -1][[0, 1, -1]],
    np.array([101.1550, 107.9061, 449.1692]), 
    decimal=2
)

# recover mse
assert math.isclose(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h)
_theta.forecast(
    init_states,
    len(ap), 
    _theta.ModelType.STM, 
    fcsts,
    alpha,
    theta,
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2418, 444.5704, 445.8990, 447.2276]),
    decimal=3
)
#optimal theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5))
mse = _theta.calc(
    ap,
    init_states, 
    _theta.ModelType.OTM, 
    initial_smoothed,
    alpha,
    theta,
    e_,
    amse_,
    3,
)
#verify we recover the fitted values
np.testing.assert_array_equal(
    ap - e_,
    init_states[:, -1]
)
#verify we get same fitted values than R
# use stm(AirPassengers, s=F, estimation=F, h = 12)
# to recover
np.testing.assert_array_almost_equal(
    init_states[:, -1][[0, 1, -1]],
    np.array([101.1550, 107.9061, 449.1692]), 
    decimal=2
)
# recover mse
assert math.isclose(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h)
_theta.forecast(
    init_states,
    len(ap), 
    _theta.ModelType.OTM, 
    fcsts,
    alpha,
    theta,
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2418, 444.5704, 445.8990, 447.2276]),
    decimal=3
)
#dynamic simple theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5))
mse = _theta.calc(
    ap,
    init_states, 
    _theta.ModelType.DSTM, 
    initial_smoothed,
    alpha,
    theta,
    e_,
    amse_,
    3,
)
#verify we recover the fitted values
np.testing.assert_array_equal(
    ap - e_,
    init_states[:, -1]
)
#verify we get same fitted values than R
# use dstm(AirPassengers, s=F, estimation=F, h = 12)
# to recover
np.testing.assert_array_almost_equal(
    init_states[:, -1][[0, 1, -1]],
    np.array([112.0000, 112.0000, 449.1805]), 
    decimal=2
)
# recover mse
assert math.isclose(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h)
_theta.forecast(
    init_states,
    len(ap), 
    _theta.ModelType.DSTM, 
    fcsts,
    alpha,
    theta
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2330, 444.5484, 445.8594, 447.1659]),
    decimal=3
)
#dynamic optimal theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5))
mse = _theta.calc(
    ap,
    init_states, 
    _theta.ModelType.DOTM, 
    initial_smoothed,
    alpha,
    theta,
    e_,
    amse_,
    3
)
#verify we recover the fitted values
np.testing.assert_array_equal(
    ap - e_,
    init_states[:, -1]
)
#verify we get same fitted values than R
# use dstm(AirPassengers, s=F, estimation=F, h = 12)
# to recover
np.testing.assert_array_almost_equal(
    init_states[:, -1][[0, 1, -1]],
    np.array([112.0000, 112.0000, 449.1805]), 
    decimal=2
)
# recover mse
assert math.isclose(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h)
_theta.forecast(
    init_states,
    len(ap), 
    _theta.ModelType.DOTM, 
    fcsts,
    alpha,
    theta
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2330, 444.5484, 445.8594, 447.1659]),
    decimal=3
)
initparamtheta(
    initial_smoothed=np.nan,
    alpha=np.nan,
    theta=np.nan,
    y=ap,
    modeltype=_theta.ModelType.DOTM,
)
switch_theta('STM')
res = thetamodel(
    y=ap,
    m=12,
    modeltype='STM',
    initial_smoothed=np.nan,
    alpha=np.nan,
    theta=np.nan, 
    nmse=3
)
forecast_theta(res, 12, level=[90, 80])
# test zero constant time series
zeros = np.zeros(30, dtype=np.float32)
res = auto_theta(zeros, m=12)
forecast_theta(res, 28)
import matplotlib.pyplot as plt
res = auto_theta(ap, m=12)
fcst = forecast_theta(res, 12, level=[80, 90])
plt.plot(np.arange(0, len(ap)), ap)
plt.plot(np.arange(len(ap), len(ap) + 12), fcst['mean'])
plt.fill_between(np.arange(len(ap), len(ap) + 12), 
                 fcst['lo-90'], 
                 fcst['hi-90'], 
                 color='orange')
res = auto_theta(ap, m=12, model='DOTM', decomposition_type='additive')
fcst = forecast_theta(res, 12, level=[80, 90])
plt.plot(np.arange(0, len(ap)), ap)
plt.plot(np.arange(len(ap), len(ap) + 12), fcst['mean'])
plt.fill_between(np.arange(len(ap), len(ap) + 12), 
                 fcst['lo-90'], 
                 fcst['hi-90'], 
                 color='orange')
# test Simple Theta Model
# with no seasonality
res = auto_theta(ap, m=1, model='STM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([432.9292, 434.2578, 435.5864, 436.9150, 438.2435]),
    fcst['mean'],
    decimal=2
)

# test Simple Theta Model
# with seasonality
res = auto_theta(ap, m=12, model='STM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([440.7886, 429.0739, 490.4933, 476.4663, 480.4363]),
    fcst['mean'],
    decimal=0
)
# test Optimized Theta Model
# with no seasonality
res = auto_theta(ap, m=1, model='OTM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([433.3307, 435.0567, 436.7828, 438.5089, 440.2350]),
    fcst['mean'],
    decimal=-1
)

# test Optimized Theta Model
# with seasonality
res = auto_theta(ap, m=12, model='OTM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([442.8492, 432.1255, 495.1706, 482.1585, 487.3280]),
    fcst['mean'],
    decimal=0
)
# test Dynamic Simple Theta Model
# with no seasonality
res = auto_theta(ap, m=1, model='DSTM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([432.9292, 434.2520, 435.5693, 436.8809, 438.1871]),
    fcst['mean'],
    decimal=2
)

# test Simple Theta Model
# with seasonality
res = auto_theta(ap, m=12, model='DSTM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([440.7631, 429.0512, 490.4711, 476.4495, 480.4251]),
    fcst['mean'],
    decimal=2
)
# test Dynamic Optimized Theta Model
# with no seasonality
res = auto_theta(ap, m=1, model='DOTM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([432.5131, 433.4257, 434.3344, 435.2391, 436.1399]),
    fcst['mean'],
    decimal=0
)

# test Simple Theta Model
# with seasonality
res = auto_theta(ap, m=12, model='DOTM')
fcst = forecast_theta(res, 5)
np.testing.assert_almost_equal(
    np.array([442.9720, 432.3586, 495.5702, 482.6789, 487.9888]),
    fcst['mean'],
    decimal=0
)
# test inttermitent time series
inttermitent_series = np.array([
    1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 3., 0., 1., 0., 0., 0., 0.,
    0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 3., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1., 1.,
    0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
    0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 3., 1., 0., 1., 0., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2.,
    1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1., 2., 0.,
    1., 0., 2., 2., 0., 0., 1., 2., 0., 0., 0., 2., 0., 1., 0., 0., 0.,
    0., 2., 0., 1., 0., 2., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 1., 1., 0.,
    0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 2.,
    1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 2., 0., 1., 0.,
    0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0.,
    1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 2., 0., 0., 0., 0., 1., 0., 1., 0., 2., 0., 0., 2., 0., 0.,
    2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 2.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
    1., 0., 1., 3., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 2., 0., 0., 1., 0., 2., 0., 0., 0., 0.,
    2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
    1., 0., 1., 0., 0., 0., 0., 3., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
    0., 0., 0., 2., 0., 1., 0., 2., 1., 2., 2., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 2., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 2., 2.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 4., 0., 0., 0., 0., 0., 1.,
    1., 0., 0., 1., 1., 0., 0., 2., 1., 1., 1., 2., 1., 0., 0., 0., 1.,
    0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 1., 2., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
    1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
    1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0.,
    1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
    0., 0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1.,
    0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
    1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
    0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1.,
    0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
    0., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)

for season_length in [1, 7]:
    res = auto_theta(inttermitent_series, m=season_length)
    fcst = forecast_theta(res, 28)
    plt.plot(np.arange(0, len(inttermitent_series)), inttermitent_series)
    plt.plot(np.arange(len(inttermitent_series), len(inttermitent_series) + 28), fcst['mean'])
    plt.show()
res = auto_theta(ap, m=12)
np.testing.assert_allclose(
    forecast_theta(forward_theta(res, ap), h=12)['mean'],
    forecast_theta(res, h=12)['mean']
)
# test tranfer
forecast_theta(forward_theta(res, inttermitent_series), h=12, level=[80,90])
res_transfer = forward_theta(res, inttermitent_series)
for key in res_transfer['par']:
    assert res['par'][key] == res_transfer['par'][key]
