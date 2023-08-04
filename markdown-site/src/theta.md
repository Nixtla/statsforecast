---
title: Theta Model
---

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import math
import os

import numpy as np
from numba import njit
from scipy.stats import norm

from statsforecast.ets import nelder_mead
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsforecast.utils import _seasonal_naive, _repeat_val_seas
```

</details>

:::

## thetacalc {#thetacalc}

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
# Global variables 
STM = 0
OTM = 1
DSTM = 2
DOTM = 3
TOL = 1.0e-10
HUGEN = 1.0e10
NA = -99999.0
smalno = np.finfo(float).eps
NOGIL = os.environ.get('NUMBA_RELEASE_GIL', 'False').lower() in ['true']
CACHE = os.environ.get('NUMBA_CACHE', 'False').lower() in ['true']
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from fastcore.test import test_eq
from statsforecast.utils import AirPassengers as ap
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def initstate(y, modeltype, initial_smoothed, alpha, theta):
    states = np.zeros((1, 5), dtype=np.float32)
    states[0, 0] = alpha * y[0] + (1 - alpha) * initial_smoothed # level
    states[0, 1] = y[0] #mean y
    if modeltype in [DSTM, DOTM]:
        # dynamic models
        states[0, 2] = y[0] # An
        states[0, 3] = 0 # Bn
        states[0, 4] = y[0] # mu
    else:
        # nodynamic models
        n = len(y)
        Bn = 6 * (2 * np.mean(np.arange(1, n + 1) * y) - (1 + n) * np.mean(y)) / ( n ** 2 - 1)
        An = np.mean(y) - (n + 1) * Bn / 2
        states[0, 2] = An
        states[0, 3] = Bn
        states[0, 4] = initial_smoothed + (1 - 1 / theta) * (An + Bn)
        
    return states
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2
initstate(ap, modeltype=STM, initial_smoothed=initial_smoothed, alpha=alpha, theta=theta)
initstate(ap, modeltype=OTM, initial_smoothed=initial_smoothed, alpha=alpha, theta=theta)
initstate(ap, modeltype=DSTM, initial_smoothed=initial_smoothed, alpha=alpha, theta=theta)
initstate(ap, modeltype=DOTM, initial_smoothed=initial_smoothed, alpha=alpha, theta=theta)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def thetacalc(y: np.ndarray,
              states: np.ndarray, # states
              modeltype: int, 
              initial_smoothed: float, 
              alpha: float,
              theta: float, 
              e: np.ndarray, 
              amse: np.ndarray, 
              nmse: int) -> float:
    denom = np.zeros(nmse)
    f = np.zeros(nmse)
    # update first state
    states[0, :] = initstate(y=y, modeltype=modeltype, 
                             initial_smoothed=initial_smoothed, 
                             alpha=alpha, theta=theta) 
    
    amse[:nmse] = 0.
    e[0] = y[0] - states[0, 4]
    n = len(y)
    for i in range(1, n):
        # one step forecast 
        thetafcst(states=states, i=i, modeltype=modeltype, f=f, h=nmse, alpha=alpha, theta=theta)
        if math.fabs(f[0] - NA) < TOL:
            mse = NA
            return mse
        e[i] = y[i] - f[0]
        for j in range(nmse):
            if (i + j) < n:
                denom[j] += 1.
                tmp = y[i + j] - f[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + (tmp * tmp)) / denom[j]
        # update state
        thetaupdate(states=states, i=i, modeltype=modeltype, 
                    alpha=alpha, theta=theta, y=y[i], usemu=0)
    mean_y = np.mean(np.abs(y))
    if math.fabs(mean_y - 0.) < TOL:
        mean_y = TOL
    mse = np.sum(e[3:] ** 2) / mean_y
    return mse
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def thetafcst(states, i, 
              modeltype, 
              f, h, 
              alpha, theta):
    # obs:
    # forecast are obtained in a recursive manner
    # this is not standard, for example in ets
    #forecasts
    new_states = np.zeros((i + h, states.shape[1]), dtype=np.float32)
    new_states[:i] = states[:i]
    for i_h in range(h):
        thetaupdate(states=new_states, i=i + i_h, modeltype=modeltype, 
                    alpha=alpha, theta=theta, y=0, usemu=1)
        f[i_h] = new_states[i + i_h, 4]  # mu is the forecast
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def thetaupdate(states, i,
                modeltype, # kind of model 
                alpha, theta,
                y, usemu):
    # states
    # level, meany, An, Bn, mu
    # get params
    level = states[i - 1, 0]
    meany = states[i - 1, 1]
    An = states[i - 1, 2]
    Bn = states[i - 1, 3]
    # update mu
    states[i, 4] = level + (1 - 1 / theta) * (An * ((1 - alpha) ** i) + Bn * (1 - (1 - alpha)**(i + 1)) / alpha)
    if usemu:
        y = states[i, 4]
    # update level
    states[i, 0] = alpha * y + (1 - alpha) * level
    # update meany
    states[i, 1] = (i * meany + y) / (i + 1)
    # update Bn and An
    if modeltype in [DSTM, DOTM]:
        # dynamic models
        states[i, 3] = ((i - 1) * Bn + 6 * (y - meany) / (i + 1)) / (i + 2)
        states[i, 2] = states[i, 1] - states[i, 3] * (i + 2) / 2
    else:
        states[i, 2] = An
        states[i, 3] = Bn
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def thetaforecast(states, n, modeltype, 
                  f, h, alpha, theta):
    # compute forecasts
    new_states = thetafcst(
        states=states, i=n, modeltype=modeltype, 
        f=f, h=h, 
        alpha=alpha,
        theta=theta
    ) 
    return new_states
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#simple theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5), dtype=np.float32)
mse = thetacalc(
    y=ap,
    states=init_states, 
    modeltype=STM, 
    initial_smoothed=initial_smoothed, alpha=alpha, theta=theta,
    e=e_, amse=amse_, nmse=3
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
test_eq(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h, dtype=np.float32)
thetaforecast(
    states=init_states, n=len(ap), 
    modeltype=STM, 
    f=fcsts, h=h, 
    alpha=alpha,
    theta=theta
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2418, 444.5704, 445.8990, 447.2276]),
    decimal=3
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#optimal theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5), dtype=np.float32)
mse = thetacalc(
    y=ap,
    states=init_states, 
    modeltype=OTM, 
    initial_smoothed=initial_smoothed, alpha=alpha, theta=theta,
    e=e_, amse=amse_, nmse=3
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
test_eq(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h, dtype=np.float32)
thetaforecast(
    states=init_states, n=len(ap), 
    modeltype=OTM, 
    f=fcsts, h=h, 
    alpha=alpha,
    theta=theta
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2418, 444.5704, 445.8990, 447.2276]),
    decimal=3
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#dynamic simple theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5), dtype=np.float32)
mse = thetacalc(
    y=ap,
    states=init_states, 
    modeltype=DSTM, 
    initial_smoothed=initial_smoothed, alpha=alpha, theta=theta,
    e=e_, amse=amse_, nmse=3
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
test_eq(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h, dtype=np.float32)
thetaforecast(
    states=init_states, n=len(ap), 
    modeltype=DSTM, 
    f=fcsts, h=h, 
    alpha=alpha,
    theta=theta
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2330, 444.5484, 445.8594, 447.1659]),
    decimal=3
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#dynamic optimal theta model tests
nmse_ = len(ap)
amse_ = np.zeros(30)
e_ = np.zeros(len(ap))
initial_smoothed = ap[0] / 2
alpha = 0.5
theta = 2.
init_states = np.zeros((len(ap), 5), dtype=np.float32)
mse = thetacalc(
    y=ap,
    states=init_states, 
    modeltype=DOTM, 
    initial_smoothed=initial_smoothed, alpha=alpha, theta=theta,
    e=e_, amse=amse_, nmse=3
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
test_eq(np.sum(e_[3:] ** 2) / np.mean(np.abs(ap)), mse)

# test forecasts
h = 5
fcsts = np.zeros(h, dtype=np.float32)
thetaforecast(
    states=init_states, n=len(ap), 
    modeltype=DOTM, 
    f=fcsts, h=h, 
    alpha=alpha,
    theta=theta
)
# test same forecast than R's
np.testing.assert_array_almost_equal(
    fcsts,
    np.array([441.9132, 443.2330, 444.5484, 445.8594, 447.1659]),
    decimal=3
)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def initparamtheta(initial_smoothed: float, alpha: float, theta: float,
                   y: np.ndarray,
                   modeltype: str):
    if modeltype in ['STM', 'DSTM']:
        if np.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = 1
        else:
            optimize_level = 0
        if np.isnan(alpha):
            alpha = 0.5
            optimize_alpha = 1
        else:
            optimize_alpha = 0
        theta = 2. # no optimize
        optimize_theta = 0
    elif modeltype in ['OTM', 'DOTM']:
        if np.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = 1
        else:
            optimize_level = 0
        if np.isnan(alpha):
            alpha = 0.5
            optimize_alpha = 1
        else:
            optimize_alpha = 0
        if np.isnan(theta):
            theta = 2.
            optimize_theta = 1
        else:
            optimize_theta = 0
    return {'initial_smoothed': initial_smoothed, 'optimize_initial_smoothed': optimize_level,
            'alpha': alpha, 'optimize_alpha': optimize_alpha,
            'theta': theta, 'optimize_theta': optimize_theta}
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
initparamtheta(initial_smoothed=np.nan, alpha=np.nan, theta=np.nan,
               y=ap,
               modeltype='DOTM')
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def switch_theta(x: str):
    return {'STM': 0, 'OTM': 1, 'DSTM': 2, 'DOTM': 3}[x]
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
switch_theta('STM')
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def pegelsresid_theta(y: np.ndarray, 
                      modeltype: str, 
                      initial_smoothed: float, alpha: float,
                      theta: float, 
                      nmse: int):
    states = np.zeros((len(y), 5), dtype=np.float32)
    e = np.full_like(y, fill_value=np.nan)
    amse = np.full(nmse, fill_value=np.nan)
    mse = thetacalc(y=y, states=states, 
                    modeltype=switch_theta(modeltype), 
                    initial_smoothed=initial_smoothed, alpha=alpha, theta=theta, 
                    e=e, amse=amse, nmse=nmse)
    if not np.isnan(mse):
        if np.abs(mse + 99999) < 1e-7:
            mse = np.nan
    return amse, e, states, mse
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def theta_target_fn(
        optimal_param,
        init_level,
        init_alpha,
        init_theta,
        opt_level,
        opt_alpha,
        opt_theta,
        y,
        modeltype,
        nmse
    ):
    states = np.zeros((len(y), 5), dtype=np.float32)
    j = 0
    if opt_level:
        level = optimal_param[j]
        j+=1
    else:
        level = init_level
        
    if opt_alpha:
        alpha = optimal_param[j]
        j+=1
    else:
        alpha = init_alpha
        
    if opt_theta:
        theta = optimal_param[j]
        j+=1
    else:
        theta = init_theta
        
    e = np.full_like(y, fill_value=np.nan)
    amse = np.full(nmse, fill_value=np.nan)
    mse = thetacalc(y=y, states=states, 
                    modeltype=switch_theta(modeltype), 
                    initial_smoothed=level, alpha=alpha, theta=theta, 
                    e=e, amse=amse, nmse=nmse)
    if mse < -1e10: 
        mse = -1e10 
    if math.isnan(mse): 
        mse = -np.inf
    if math.fabs(mse + 99999) < 1e-7: 
        mse = -np.inf
    return mse
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def optimize_theta_target_fn(
        init_par, optimize_params, y, 
        modeltype, nmse
    ):
    x0 = [init_par[key] for key, val in optimize_params.items() if val]
    x0 = np.array(x0, dtype=np.float32)
    if not len(x0):
        return
    
    init_level = init_par['initial_smoothed']
    init_alpha = init_par['alpha']
    init_theta = init_par['theta']
    
    opt_level = optimize_params['initial_smoothed']
    opt_alpha = optimize_params['alpha']
    opt_theta = optimize_params['theta']
    
    res = nelder_mead(
        theta_target_fn, x0, 
        args=(
            init_level,
            init_alpha,
            init_theta,
            opt_level,
            opt_alpha,
            opt_theta,
            y,
            modeltype,
            nmse
        ),
        tol_std=1e-4, 
        lower=np.array([-1e10, 0.1, 1.0]),
        upper=np.array([1e10, 0.99, 1e10]),
        max_iter=1_000,
        adaptive=True,
    )
    return res
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
@njit(nogil=NOGIL, cache=CACHE)
def is_constant(x):
    return np.all(x[0] == x)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
is_constant(ap)
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def thetamodel(
        y: np.ndarray, m: int, 
        modeltype: str, 
        initial_smoothed: float, alpha: float,
        theta: float, nmse: int
    ):
    #initial parameters
    par = initparamtheta(initial_smoothed=initial_smoothed, 
                         alpha=alpha, theta=theta, 
                         y=y, modeltype=modeltype)
    optimize_params = {key.replace('optimize_', ''): val for key, val in par.items() if 'optim' in key}
    par = {key: val for key, val in par.items() if 'optim' not in key}
    # parameter optimization
    fred = optimize_theta_target_fn(
        init_par=par, optimize_params=optimize_params, y=y, 
        modeltype=modeltype, nmse=nmse
    )
    if fred is not None:
        fit_par = fred.x
    j = 0
    if optimize_params['initial_smoothed']:
        j += 1
    if optimize_params['alpha']:
        par['alpha'] = fit_par[j]
        j += 1
    if optimize_params['theta']:
        par['theta'] = fit_par[j]
        j += 1
    
    amse, e, states, mse = pegelsresid_theta(
        y=y, modeltype=modeltype,
        nmse=nmse, **par
    )
    
    return dict(mse=mse, amse=amse, fit=fred, residuals=e,
                m=m, states=states, par=par, n=len(y), 
                modeltype=modeltype, mean_y=np.mean(y))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
res = thetamodel(
    y=ap, m=12, modeltype='STM',
    initial_smoothed=np.nan,
    alpha=np.nan,
    theta=np.nan, 
    nmse=3
)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def compute_pi_samples(n, h, states, sigma, alpha, theta, mean_y, seed=0, n_samples=200):
    samples = np.full((h, n_samples), fill_value=np.nan, dtype=np.float32)
    # states: level, meany, An, Bn, mu
    smoothed, _, A, B, _ = states[-1]
    np.random.seed(seed)
    for i in range(n, n + h):
        samples[i - n] = smoothed + (1 - 1 / theta)*(A*((1 - alpha) ** i) + B * (1 - (1 - alpha)**(i + 1)) / alpha)
        samples[i - n] += np.random.normal(scale=sigma, size=n_samples)
        smoothed = alpha * samples[i - n] + (1 - alpha) * smoothed
        mean_y = (i * mean_y + samples[i - n]) / (i + 1)
        B = ((i - 1) * B + 6 * (samples[i - n] - mean_y) / (i + 1)) / (i + 2)
        A = mean_y - B * (i + 2) / 2
    return samples
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def forecast_theta(obj, h, level=None):
    forecast = np.full(h, fill_value=np.nan)
    n = obj['n']
    states = obj['states']
    alpha=obj['par']['alpha']
    theta=obj['par']['theta']
    thetaforecast(
        states=states, n=n, modeltype=switch_theta(obj['modeltype']), 
        h=h, f=forecast, alpha=alpha, theta=theta
    )
    res = {'mean': forecast}
    
    if level is not None:
        sigma = np.std(obj['residuals'][3:], ddof=1)
        mean_y = obj['mean_y']
        samples = compute_pi_samples(n=n, h=h, states=states, sigma=sigma, alpha=alpha, 
                                     theta=theta, mean_y=mean_y)
        for lv in level:
            min_q = (100 - lv) / 200
            max_q = min_q + lv / 100
            res[f'lo-{lv}'] = np.quantile(samples, min_q, axis=1)
            res[f'hi-{lv}'] = np.quantile(samples, max_q, axis=1)
            
    if obj.get('decompose', False):
        seas_forecast = _repeat_val_seas(obj['seas_forecast']['mean'], h=h, season_length=obj['m'])
        for key in res:
            if obj['decomposition_type'] == 'multiplicative':
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast
    return res
```

</details>

:::

<details>
<summary>Code</summary>

``` python
forecast_theta(res, 12, level=[90, 80])
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def auto_theta(
        y, m, model=None, 
        initial_smoothed=None, alpha=None, 
        theta=None,
        nmse=3,
        decomposition_type='multiplicative'
    ):
    # converting params to floats 
    # to improve numba compilation
    if initial_smoothed is None:
        initial_smoothed = np.nan
    if alpha is None:
        alpha = np.nan
    if theta is None:
        theta = np.nan
    if nmse < 1 or nmse > 30:
        raise ValueError('nmse out of range')
    # constan values
    if is_constant(y):
        thetamodel(y=y, m=m, modeltype='STM', nmse=nmse, 
                  initial_smoothed=np.mean(y) / 2, alpha=0.5, theta=2.0)
    # seasonal decomposition if needed
    decompose = False
    # seasonal test
    if m >= 4:
        r = acf(y, nlags=m, fft=False)[1:]
        stat = np.sqrt((1 + 2 * np.sum(r[:-1]**2)) / len(y))
        decompose = np.abs(r[-1]) / stat > norm.ppf(0.95)
    
    data_positive = min(y) > 0
    if decompose:
        # change decomposition type if data is not positive
        if decomposition_type == 'multiplicative' and not data_positive:
            decomposition_type = 'additive'
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m).seasonal
        if decomposition_type == 'multiplicative' and any(y_decompose < 0.01):
            decomposition_type = 'additive'
            y_decompose = seasonal_decompose(y, model='additive', period=m).seasonal
        if decomposition_type == 'additive':
            y = y - y_decompose
        else:
            y = y / y_decompose
        seas_forecast = _seasonal_naive(y=y_decompose, h=m, season_length=m, fitted=False)
    
    # validate model
    if model not in [None, 'STM', 'OTM', 'DSTM', 'DOTM']:
        raise ValueError('Invalid model type')

    n = len(y)
    npars = 3 
    #non-optimized tiny datasets
    if n <= npars:
        raise NotImplementedError('tiny datasets')
    if model is None:
        modeltype = ['STM', 'OTM', 'DSTM', 'DOTM']
    else:
        modeltype = [model]
        
    best_ic = np.inf
    for mtype in modeltype:
        fit = thetamodel(y=y, m=m, modeltype=mtype, nmse=nmse, 
                         initial_smoothed=initial_smoothed, alpha=alpha, theta=theta)
        fit_ic = fit['mse']
        if not np.isnan(fit_ic):
            if fit_ic < best_ic:
                model = fit
                best_ic = fit_ic
    if np.isinf(best_ic):
        raise Exception('no model able to be fitted')
        
    if decompose:
        if decomposition_type == 'multiplicative':
            model['residuals'] = model['residuals'] * y_decompose
        else:
            model['residuals'] = model['residuals'] + y_decompose
        model['decompose'] = decompose
        model['decomposition_type'] = decomposition_type
        model['seas_forecast'] = dict(seas_forecast)
    return model
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test zero constant time series
zeros = np.zeros(30, dtype=np.float32)
res = auto_theta(zeros, m=12)
forecast_theta(res, 28)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt
res = auto_theta(ap, m=12)
fcst = forecast_theta(res, 12, level=[80, 90])
plt.plot(np.arange(0, len(ap)), ap)
plt.plot(np.arange(len(ap), len(ap) + 12), fcst['mean'])
plt.fill_between(np.arange(len(ap), len(ap) + 12), 
                 fcst['lo-90'], 
                 fcst['hi-90'], 
                 color='orange')
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt
res = auto_theta(ap, m=12, model='DOTM', decomposition_type='additive')
fcst = forecast_theta(res, 12, level=[80, 90])
plt.plot(np.arange(0, len(ap)), ap)
plt.plot(np.arange(len(ap), len(ap) + 12), fcst['mean'])
plt.fill_between(np.arange(len(ap), len(ap) + 12), 
                 fcst['lo-90'], 
                 fcst['hi-90'], 
                 color='orange')
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
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
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
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
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
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
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
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
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
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

import matplotlib.pyplot as plt
for season_length in [1, 7]:
    res = auto_theta(inttermitent_series, m=season_length)
    fcst = forecast_theta(res, 28)
    plt.plot(np.arange(0, len(inttermitent_series)), inttermitent_series)
    plt.plot(np.arange(len(inttermitent_series), len(inttermitent_series) + 28), fcst['mean'])
    plt.show()
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def forward_theta(fitted_model, y):
    m = fitted_model['m']
    model = fitted_model['modeltype']
    initial_smoothed = fitted_model['par']['initial_smoothed']
    alpha = fitted_model['par']['alpha']
    theta = fitted_model['par']['theta']
    return auto_theta(y=y, m=m, model=model, 
                      initial_smoothed=initial_smoothed, 
                      alpha=alpha, 
                      theta=theta)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
res = auto_theta(ap, m=12)
test_eq(
    forecast_theta(forward_theta(res, ap), h=12)['mean'],
    forecast_theta(res, h=12)['mean']
)
# test tranfer
forecast_theta(forward_theta(res, inttermitent_series), h=12, level=[80,90])
res_transfer = forward_theta(res, inttermitent_series)
for key in res_transfer['par']:
    test_eq(res['par'][key], res_transfer['par'][key])
```

</details>

:::

