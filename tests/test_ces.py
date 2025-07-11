from fastcore.test import test_eq
from statsforecast.utils import AirPassengers as ap
initstate(ap, 12, 'N')
initstate(ap, 12, 'S')
initstate(ap, 12, 'P')
initstate(ap, 12, 'F')
#nonseasonal test
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
alpha_0 = 2.001457
alpha_1 = 1.000727
beta_0 = 0.
beta_1 = 0.
init_states_non_seas = np.zeros((2 + len(ap), 2), dtype=np.float32)
init_states_non_seas[0] = initstate(ap, 12, 'N')
cescalc(y=ap,
        states=init_states_non_seas, m=12, 
        season=NONE, alpha_0=alpha_0, 
        alpha_1=alpha_1, beta_0=beta_0, 
        beta_1=beta_1,
        e=e_, amse=amse_, nmse=3, 
        backfit=1)
np.testing.assert_array_equal(
    init_states_non_seas[[0, -2, -1]],
    np.array([
        [  112.06887, 1301.9882 ],
        [  430.92154 , 2040.1951 ],
        [  432.40475, -1612.2461 ]
    ], dtype=np.float32)
)
#nonseasonal forecast test
h = 13
fcsts = np.zeros(h, dtype=np.float32)
cesforecast(states=init_states_non_seas, n=len(ap), m=12, 
            season=NONE, 
            f=fcsts, h=h, 
            alpha_0=alpha_0, alpha_1=alpha_1, 
            beta_0=beta_0, beta_1=beta_1)
#taken from R using ces(AirPassengers, h=13)
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([
        430.9211, 432.4049, 431.2324, 432.7212, 431.5439,
        433.0376, 431.8556, 433.3543, 432.1675, 433.6712,
        432.4796, 433.9884, 432.7920
    ], dtype=np.float32), 
    decimal=2
)
#simple seasonal test
nmse_ = len(ap)
amse_ = np.zeros(30)
lik_ = 0.
e_ = np.zeros(len(ap))
alpha_0 = 1.996411
alpha_1 = 1.206694
beta_0 = 0.
beta_1 = 0.
m = 12
init_states_s_ses = np.zeros((12 * 2 + len(ap), 2), dtype=np.float32)
init_states_s_ses[:m] = initstate(ap, m, 'S')
cescalc(y=ap, 
        states=init_states_s_ses, m=12, 
        season=SIMPLE, alpha_0=alpha_0, 
        alpha_1=alpha_1, beta_0=beta_0, 
        beta_1=beta_1,
        e=e_, amse=amse_, nmse=3, backfit=1)
np.testing.assert_array_equal(
    init_states_s_ses[[0, 11, 145, 143 + 12]],
    np.array([
        [130.49458 ,  36.591137],
        [135.21922 , 121.62022 ],
        [423.57788 , 252.81241 ],
        [505.3621  ,  95.29781 ]
    ], dtype=np.float32)
)
#simple seasonal forecast test
h = 13
fcsts = np.zeros(h, dtype=np.float32)
cesforecast(states=init_states_s_ses, n=len(ap), m=12, 
            season=SIMPLE, 
            f=fcsts, h=h, 
            alpha_0=alpha_0, alpha_1=alpha_1, 
            beta_0=beta_0, beta_1=beta_1)
#taken from R using ces(AirPassengers, h=13, seasonality = 'simple')
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([
        446.2768, 423.5779, 481.4365, 514.7730, 533.5008,
        589.0500, 688.2703, 674.5891, 580.9486, 516.0776,
        449.7246, 505.3621, 507.9884
    ], dtype=np.float32), 
    decimal=2
)
#partial seasonal test
nmse_ = len(ap)
amse_ = np.zeros(30)
lik_ = 0.
e_ = np.zeros(len(ap))
alpha_0 = 1.476837
alpha_1 = 1.
beta_0 = 0.91997
beta_1 = 0.
m = 12
init_states_p_seas = np.zeros((12 + len(ap), 3), dtype=np.float32)
init_states_p_seas[:m] = initstate(ap, m, 'P')
cescalc(y=ap, 
        states=init_states_p_seas, m=12, 
        season=2, alpha_0=alpha_0, 
        alpha_1=alpha_1, beta_0=beta_0, 
        beta_1=beta_1,
        e=e_, amse=amse_, nmse=3, backfit=1)
np.testing.assert_array_equal(
    init_states_p_seas[[0, 11, 145, 143 + 12]],
    np.array([
        [122.580666,  83.00358 ,  -9.710966],
        [122.580666,  78.11936 ,  -4.655848],
        [438.5037  , 300.70374 , -25.55726 ],
        [438.5037  , 296.92316 ,  -7.581563]
    ], dtype=np.float32)
)
#partial seasonal forecast test
h = 13
fcsts = np.zeros(h, dtype=np.float32)
cesforecast(states=init_states_p_seas, n=len(ap), m=12, 
            season=PARTIAL, 
            f=fcsts, h=h, 
            alpha_0=alpha_0, alpha_1=alpha_1, 
            beta_0=beta_0, beta_1=beta_1)
#taken from R using ces(AirPassengers, h=13, seasonality = 'partial')
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([
        437.6247, 412.9464, 445.5811, 498.5370, 493.0405, 550.7443, 
        629.2205, 607.1793, 512.3455, 462.1260, 383.4097, 430.9221, 437.6247
    ], dtype=np.float32), 
    decimal=2
)
#full seasonal test
nmse_ = len(ap)
amse_ = np.zeros(30)
lik_ = 0.
e_ = np.zeros(len(ap))
alpha_0 = 1.350795
alpha_1 = 1.009169
beta_0 = 1.777909
beta_1 = 0.973739
m = 12
init_states_f_seas = np.zeros((12 * 2 + len(ap), 4), dtype=np.float32)
init_states_f_seas[:m] = initstate(ap, m, 'F')
cescalc(y=ap,
        states=init_states_f_seas, m=12, 
        season=3, alpha_0=alpha_0, 
        alpha_1=alpha_1, beta_0=beta_0, 
        beta_1=beta_1,
        e=e_, amse=amse_, nmse=3, backfit=1)
np.testing.assert_array_equal(
    init_states_f_seas[[0, 11, 145, 143 + 12]],
    np.array([
        [ 227.74284 ,  167.7603  ,  -94.299805,  -39.623283],
        [ 211.48921 ,  155.72342 ,  -91.62251 ,  -82.953064],
        [ 533.1726  ,  372.95758 , -139.31824 , -125.856834],
        [ 564.9041  ,  404.3251  , -130.9048  , -137.33    ]
    ], dtype=np.float32)
)
#full seasonal forecast test
h = 13
fcsts = np.zeros(h, dtype=np.float32)
cesforecast(states=init_states_f_seas, n=len(ap), m=12, 
            season=FULL, 
            f=fcsts, h=h, 
            alpha_0=alpha_0, alpha_1=alpha_1, 
            beta_0=beta_0, beta_1=beta_1)
#taken from R using ces(AirPassengers, h=13, seasonality = 'full')
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([
        450.9262, 429.2925, 465.4771, 510.1799, 517.9913, 578.5654,
        655.9219, 638.6218, 542.0985, 498.1064, 431.3293, 477.3273,
        501.3757
    ], dtype=np.float32), 
    decimal=2
)
initparamces(alpha_0=np.nan, alpha_1=np.nan, 
             beta_0=np.nan, beta_1=np.nan, 
             seasontype='N')
switch_ces('N')
res = cesmodel(
    y=ap, m=12, seasontype='N',
    alpha_0=np.nan,
    alpha_1=np.nan,
    beta_0=np.nan, 
    beta_1=np.nan,
    nmse=3
)
forecast_ces(res, 12)
import matplotlib.pyplot as plt
res = auto_ces(ap, m=12, model='F')
fcst = forecast_ces(res, 12)
plt.plot(np.arange(0, len(ap)), ap)
plt.plot(np.arange(len(ap), len(ap) + 12), fcst['mean'])
res = auto_ces(ap, m=12)
test_eq(
    forecast_ces(forward_ces(res, ap), h=12)['mean'],
    forecast_ces(res, h=12)['mean']
)
# test tranfer
forecast_ces(forward_ces(res, np.log(ap)), h=12)
res_transfer = forward_ces(res, np.log(ap))
for key in res_transfer['par']:
    test_eq(res['par'][key], res_transfer['par'][key])
# less than two seasonal periods removes seasonal component
res = auto_ces(np.arange(23, dtype=np.float64), m=12)
assert res['seasontype'] == 'N'
