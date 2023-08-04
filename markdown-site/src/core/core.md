---
title: Core Methods
---

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
%load_ext autoreload
%autoreload 2
```

</details>

:::

> Methods for Fit, Predict, Forecast (fast), Cross Validation and
> plotting

The core methods of `StatsForecast` are:

-   `StatsForecast.fit`
-   `StatsForecast.predict`
-   `StatsForecast.forecast`
-   `StatsForecast.cross_validation`
-   `StatsForecast.plot`

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from nbdev.showdoc import add_docs, show_doc
from statsforecast.models import Naive
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import inspect
import logging
import random
import re
from itertools import product
from os import cpu_count
from typing import Any, List, Optional, Union, Dict
import pkg_resources

from fugue.execution.factory import make_execution_engine
import matplotlib.pyplot as plt
import matplotlib.colors as cm              
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go    
from plotly.subplots import make_subplots
from tqdm.autonotebook import tqdm
from triad import conditional_dispatcher
from fugue.execution.factory import try_get_context_execution_engine

from statsforecast.utils import ConformalIntervals
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
logger = logging.getLogger(__name__)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
logger.setLevel(logging.ERROR)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from fastcore.test import test_eq, test_fail
from statsforecast.utils import generate_series
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
class GroupedArray:
    
    def __init__(self, data, indptr):
        self.data = data
        self.indptr = indptr
        self.n_groups = self.indptr.size - 1
        
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[self.indptr[idx] : self.indptr[idx + 1]]
        elif isinstance(idx, slice):
            idx = slice(idx.start, idx.stop + 1, idx.step)
            new_indptr = self.indptr[idx].copy()
            new_data = self.data[new_indptr[0] : new_indptr[-1]].copy()            
            new_indptr -= new_indptr[0]
            return GroupedArray(new_data, new_indptr)
        raise ValueError(f'idx must be either int or slice, got {type(idx)}')
    
    def __len__(self):
        return self.n_groups
    
    def __repr__(self):
        return f'GroupedArray(n_data={self.data.size:,}, n_groups={self.n_groups:,})'
    
    def __eq__(self, other):
        if not hasattr(other, 'data') or not hasattr(other, 'indptr'):
            return False
        return np.allclose(self.data, other.data) and np.array_equal(self.indptr, other.indptr)
    
    def fit(self, models):
        fm = np.full((self.n_groups, len(models)), np.nan, dtype=object)
        for i, grp in enumerate(self):
            y = grp[:, 0] if grp.ndim == 2 else grp
            X = grp[:, 1:] if (grp.ndim == 2 and grp.shape[1] > 1) else None
            for i_model, model in enumerate(models):
                new_model = model.new()
                fm[i, i_model] = new_model.fit(y=y, X=X)
        return fm
    
    def _get_cols(self, models, attr, h, X, level=tuple()):
        n_models = len(models)
        cuts = np.full(n_models + 1, fill_value=0, dtype=np.int32)
        has_level_models = np.full(n_models, fill_value=False, dtype=bool) 
        cuts[0] = 0
        for i_model, model in enumerate(models):
            len_cols = 1 # mean
            has_level = 'level' in inspect.signature(getattr(model, attr)).parameters and len(level) > 0
            has_level_models[i_model] = has_level
            if has_level:
                len_cols += 2 * len(level) #levels
            cuts[i_model + 1] = len_cols + cuts[i_model]
        return cuts, has_level_models
    
    def _output_fcst(self, models, attr, h, X, level=tuple()):
        #returns empty output according to method
        cuts, has_level_models = self._get_cols(models=models, attr=attr, h=h, X=X, level=level)
        out = np.full((self.n_groups * h, cuts[-1]), fill_value=np.nan, dtype=np.float32)
        return out, cuts, has_level_models
        
    def predict(self, fm, h, X=None, level=tuple()):
        #fm stands for fitted_models
        #and fm should have fitted_model
        fcsts, cuts, has_level_models = self._output_fcst(
            models=fm[0], attr='predict', 
            h=h, X=X, level=level
        )
        matches = ['mean', 'lo', 'hi']
        cols = []
        for i_model in range(fm.shape[1]):
            has_level = has_level_models[i_model]
            kwargs = {}
            if has_level:
                kwargs['level'] = level
            for i, _ in enumerate(self):
                if X is not None:
                    X_ = X[i]
                else:
                    X_ = None
                res_i = fm[i, i_model].predict(h=h, X=X_, **kwargs)
                cols_m = [key for key in res_i.keys() if any(key.startswith(m) for m in matches)]
                fcsts_i = np.vstack([res_i[key] for key in cols_m]).T
                model_name = repr(fm[i, i_model])
                cols_m = [f'{model_name}' if col == 'mean' else f'{model_name}-{col}' for col in cols_m]
                if fcsts_i.ndim == 1:
                    fcsts_i = fcsts_i[:, None]
                fcsts[i * h : (i + 1) * h, cuts[i_model]:cuts[i_model + 1]] = fcsts_i
            cols += cols_m
        return fcsts, cols
    
    def fit_predict(self, models, h, X=None, level=tuple()):
        #fitted models
        fm = self.fit(models=models)
        #forecasts
        fcsts, cols = self.predict(fm=fm, h=h, X=X, level=level)
        return fm, fcsts, cols
    
    def forecast(self, models, h, fallback_model=None, fitted=False, X=None, level=tuple(), verbose=False):
        fcsts, cuts, has_level_models = self._output_fcst(
            models=models, attr='forecast', 
            h=h, X=X, level=level
        )
        matches = ['mean', 'lo', 'hi']
        matches_fitted = ['fitted', 'fitted-lo', 'fitted-hi']
        if fitted:
            #for the moment we dont return levels for fitted values in 
            #forecast mode
            fitted_vals = np.full((self.data.shape[0], 1 + cuts[-1]), np.nan, dtype=np.float32)
            if self.data.ndim == 1:
                fitted_vals[:, 0] = self.data
            else:
                fitted_vals[:, 0] = self.data[:, 0]
        iterable = tqdm(enumerate(self), 
                        disable=(not verbose), 
                        total=len(self),
                        desc='Forecast')
        for i, grp in iterable:
            y_train = grp[:, 0] if grp.ndim == 2 else grp
            X_train = grp[:, 1:] if (grp.ndim == 2 and grp.shape[1] > 1) else None
            if X is not None:
                X_f = X[i]
            else:
                X_f = None
            cols = []
            cols_fitted = []
            for i_model, model in enumerate(models):
                has_level = has_level_models[i_model]
                kwargs = {}
                if has_level:
                    kwargs['level'] = level
                try:
                    res_i = model.forecast(h=h, y=y_train, X=X_train, X_future=X_f, fitted=fitted, **kwargs)
                except Exception as error:
                    if fallback_model is not None:
                        res_i = fallback_model.forecast(h=h, y=y_train, X=X_train, X_future=X_f, fitted=fitted, **kwargs)
                    else:
                        raise error
                cols_m = [key for key in res_i.keys() if any(key.startswith(m) for m in matches)]
                fcsts_i = np.vstack([res_i[key] for key in cols_m]).T
                cols_m = [f'{repr(model)}' if col == 'mean' else f'{repr(model)}-{col}' for col in cols_m]
                if fcsts_i.ndim == 1:
                    fcsts_i = fcsts_i[:, None]
                fcsts[i * h : (i + 1) * h, cuts[i_model]:cuts[i_model + 1]] = fcsts_i
                cols += cols_m
                if fitted:
                    cols_m_fitted = [key for key in res_i.keys() if any(key.startswith(m) for m in matches_fitted)]
                    fitted_i = np.vstack([res_i[key] for key in cols_m_fitted]).T
                    cols_m_fitted = [f'{repr(model)}' \
                                     if col == 'fitted' else f"{repr(model)}-{col.replace('fitted-', '')}" \
                                     for col in cols_m_fitted]
                    fitted_vals[self.indptr[i] : self.indptr[i + 1], (cuts[i_model] + 1):(cuts[i_model + 1] + 1)] = fitted_i
                    cols_fitted += cols_m_fitted
        result = {'forecasts': fcsts, 'cols': cols}
        if fitted:
            result['fitted'] = {'values': fitted_vals}
            result['fitted']['cols'] = ['y'] + cols_fitted
        return result
    
    def cross_validation(self, models, h, test_size, fallback_model=None,
                         step_size=1, input_size=None, fitted=False, level=tuple(), 
                         refit=True, verbose=False):
        # output of size: (ts, window, h)
        if (test_size - h) % step_size:
            raise Exception('`test_size - h` should be module `step_size`')
        n_windows = int((test_size - h) / step_size) + 1
        n_models = len(models)
        cuts, has_level_models = self._get_cols(models=models, attr='forecast', h=h, X=None, level=level)
        # first column of out is the actual y
        out = np.full((self.n_groups, n_windows, h, 1 + cuts[-1]), np.nan, dtype=np.float32)
        if fitted:
            fitted_vals = np.full((self.data.shape[0], n_windows, n_models + 1), np.nan, dtype=np.float32)
            fitted_idxs = np.full((self.data.shape[0], n_windows), False, dtype=bool)
            last_fitted_idxs = np.full_like(fitted_idxs, False, dtype=bool)
        matches = ['mean', 'lo', 'hi']
        steps = list(range(-test_size, -h + 1, step_size))
        for i_ts, grp in enumerate(self):
            iterable = tqdm(enumerate(steps, start=0), 
                            desc=f'Cross Validation Time Series {i_ts + 1}', 
                            disable=(not verbose),
                            total=len(steps))
            for i_window, cutoff in iterable:
                end_cutoff = cutoff + h
                in_size_disp = cutoff if input_size is None else input_size 
                y = grp[(cutoff - in_size_disp):cutoff]
                y_train = y[:, 0] if y.ndim == 2 else y
                X_train = y[:, 1:] if (y.ndim == 2 and y.shape[1] > 1) else None
                y_test = grp[cutoff:] if end_cutoff == 0 else grp[cutoff:end_cutoff]
                X_future = y_test[:, 1:] if (y_test.ndim == 2 and y_test.shape[1] > 1) else None
                out[i_ts, i_window, :, 0] = y_test[:, 0] if y.ndim == 2 else y_test
                if fitted:
                    fitted_vals[self.indptr[i_ts] : self.indptr[i_ts + 1], i_window, 0][
                        (cutoff - in_size_disp):cutoff
                    ] = y_train
                    fitted_idxs[self.indptr[i_ts] : self.indptr[i_ts + 1], i_window][
                        (cutoff - in_size_disp):cutoff
                    ] = True
                    last_fitted_idxs[
                        self.indptr[i_ts] : self.indptr[i_ts + 1], i_window
                    ][cutoff-1] = True
                cols = ['y']
                for i_model, model in enumerate(models):
                    has_level = has_level_models[i_model]
                    kwargs = {}
                    if has_level:
                        kwargs['level'] = level
                    if refit:
                        try:
                            res_i = model.forecast(h=h, y=y_train, X=X_train, 
                                                   X_future=X_future, fitted=fitted, **kwargs)
                        except Exception as error:
                            if fallback_model is not None:
                                res_i = fallback_model.forecast(h=h, y=y_train, X=X_train, 
                                                                X_future=X_future, fitted=fitted, **kwargs)
                            else:
                                raise error
                    else:
                        if i_window == 0:
                            # for the first window we have to fit each model
                            model = model.fit(y=y_train, X=X_train)
                            if fallback_model is not None:
                                fallback_model = fallback_model.fit(y=y_train, X=X_train)
                        try:
                            res_i = model.forward(h=h, y=y_train, X=X_train, 
                                                                   X_future=X_future, fitted=fitted, **kwargs)
                        except Exception as error:
                            if fallback_model is not None:
                                res_i = fallback_model.forward(h=h, y=y_train, X=X_train, 
                                                               X_future=X_future, fitted=fitted, **kwargs)
                            else:
                                raise error
                    cols_m = [key for key in res_i.keys() if any(key.startswith(m) for m in matches)]
                    fcsts_i = np.vstack([res_i[key] for key in cols_m]).T
                    cols_m = [f'{repr(model)}' if col == 'mean' else f'{repr(model)}-{col}' for col in cols_m]
                    out[i_ts, i_window, :, (1 + cuts[i_model]):(1 + cuts[i_model + 1])] = fcsts_i
                    if fitted:
                        fitted_vals[self.indptr[i_ts] : self.indptr[i_ts + 1], i_window, i_model + 1][
                            (cutoff - in_size_disp):cutoff
                        ] = res_i['fitted']
                    cols += cols_m
        result = {'forecasts': out.reshape(-1, 1 + cuts[-1]), 'cols': cols}
        if fitted:
            result['fitted'] = {
                'values': fitted_vals, 
                'idxs': fitted_idxs, 
                'last_idxs': last_fitted_idxs,
                'cols': ['y'] + [repr(model) for model in models]
            }
        return result

    def split(self, n_chunks):
        return [self[x[0] : x[-1] + 1] for x in np.array_split(range(self.n_groups), n_chunks) if x.size]
    
    def split_fm(self, fm, n_chunks):
        return [fm[x[0] : x[-1] + 1] for x in np.array_split(range(self.n_groups), n_chunks) if x.size]
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# sum ahead just returns the last value
# added with h future values 
class SumAhead:
    
    def __init__(self):
        pass
    
    def fit(self, y, X):
        self.last_value = y[-1]
        self.fitted_values = np.full(y.size, np.nan, np.float32)
        self.fitted_values[1:] = y[:1]
        return self
    
    def predict(self, h, X=None, level=None):
        mean = self.last_value + np.arange(1, h + 1)
        res = {'mean': mean}
        if level is not None:
            for lv in level:
                res[f'lo-{lv}'] = mean - 1.0
                res[f'hi-{lv}'] = mean + 1.0
        return res
    
    def __repr__(self):
        return 'SumAhead'
    
    def forecast(self, y, h, X=None, X_future=None, fitted=False, level=None):
        mean = y[-1] + np.arange(1, h + 1)
        res = {'mean': mean}
        if fitted:
            fitted_values = np.full(y.size, np.nan, np.float32)
            fitted_values[1:] = y[1:]
            res['fitted'] = fitted_values
        if level is not None:
            for lv in level:
                res[f'lo-{lv}'] = mean - 1.0
                res[f'hi-{lv}'] = mean + 1.0
        return res
    
    def forward(self, y, h, X=None, X_future=None, fitted=False, level=None):
        # fix self.last_value for test purposes
        mean = self.last_value + np.arange(1, h + 1)
        res = {'mean': mean}
        if fitted:
            fitted_values = np.full(y.size, np.nan, np.float32)
            fitted_values[1:] = y[1:]
            res['fitted'] = fitted_values
        if level is not None:
            for lv in level:
                res[f'lo-{lv}'] = mean - 1.0
                res[f'hi-{lv}'] = mean + 1.0
        return res
    
    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#data used for tests
data = np.arange(12)
indptr = np.array([0, 4, 8, 12])

# test we can recover the 
# number of series
ga = GroupedArray(data, indptr)
test_eq(len(ga), 3)

#test splits of data
splits = ga.split(2)
test_eq(splits[0], GroupedArray(data[:8], indptr[:3]))
test_eq(splits[1], GroupedArray(data[8:], np.array([0, 4])))

# fitting models for each ts
models = [Naive(), Naive()]
fm = ga.fit(models)
test_eq(fm.shape, (3, 2))
test_eq(len(ga.split_fm(fm, 2)), 2)

# test forecasts
exp_fcsts = np.hstack([2 * [data[i]] for i in indptr[1:] - 1])
fcsts, cols = ga.predict(fm=fm, h=2)
np.testing.assert_equal(
    fcsts,
    np.hstack([exp_fcsts[:, None], exp_fcsts[:, None]]),
)

#test fit and predict pipelie
fm_fp, fcsts_fp, cols_fp = ga.fit_predict(models=models, h=2) 
test_eq(fm_fp.shape, (3, 2))
np.testing.assert_equal(fcsts_fp, fcsts)
np.testing.assert_equal(cols_fp, cols)

#test levels
fm_lv, fcsts_lv, cols_lv = ga.fit_predict(models=models, h=2, level=(50, 90))
test_eq(fcsts_lv.shape, (2 * len(ga), 10)) 

#test forecast
fcst_f = ga.forecast(models=models, h=2, fitted=True)
test_eq(fcst_f['forecasts'], fcsts_fp)
test_eq(fcst_f['cols'], cols_fp)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
class NullModel:
    
    def __init__(self):
        pass
    
    def forecast(self):
        pass
    
    def __repr__(self):
        return "NullModel"

#test fallback model
fcst_f = ga.forecast(models=[NullModel(), NullModel()], fallback_model=Naive(), h=2, fitted=True)
test_eq(fcst_f['forecasts'], fcsts_fp)
test_eq(fcst_f['cols'], ['NullModel', 'NullModel'])
test_fail(ga.forecast, kwargs={'models': [NullModel()]})
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#test levels
lv = (50, 60)
h = 2
#test for forecasts
fcsts_lv = ga.forecast(models=[SumAhead()], h=h, fitted=True, level=lv)
test_eq(
    fcsts_lv['forecasts'].shape,
    (len(ga) * h, 1 + 2 * len(lv))
)
test_eq(
    fcsts_lv['cols'],
    ['SumAhead', 
     'SumAhead-lo-50', 
     'SumAhead-hi-50',
     'SumAhead-lo-60',
     'SumAhead-hi-60']
)
#fit and predict pipeline
fm_lv_fp, fcsts_lv_fp, cols_lv_fp = ga.fit_predict(models=[SumAhead()], h=h, level=lv)
test_eq(
    fcsts_lv['forecasts'],
    fcsts_lv_fp
)
test_eq(
    fcsts_lv['cols'],
    cols_lv_fp
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# tests for cross valiation
data = np.hstack([np.arange(10), np.arange(100, 200), np.arange(20, 40)])
indptr = np.array([0, 10, 110, 130])
ga = GroupedArray(data, indptr)
    
res_cv = ga.cross_validation(models=[SumAhead()], h=2, test_size=5, fitted=True)
fcsts_cv = res_cv['forecasts']
cols_cv = res_cv['cols']
test_eq(
    fcsts_cv[:, cols_cv.index('y')], 
    fcsts_cv[:, cols_cv.index('SumAhead')]
)

#levels
res_cv_lv = ga.cross_validation(models=[SumAhead(), Naive()], h=2, test_size=5, level=(50, 60))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
actual_step_size = np.unique(np.diff(fcsts_cv[:, cols_cv.index('SumAhead')].reshape((3, -1, 2)), axis=1))
test_eq(actual_step_size, 1)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
horizons = [1, 2, 3, 2]
test_sizes = [3, 4, 6, 6]
step_sizes = [2, 2, 3, 4]
for h, test_size, step_size in zip(horizons, test_sizes, step_sizes):
    res_cv = ga.cross_validation(
        models=[SumAhead()], h=h, 
        test_size=test_size, 
        step_size=step_size,
        fitted=True
    )
    fcsts_cv = res_cv['forecasts']
    cols_cv = res_cv['cols']
    test_eq(
        fcsts_cv[:, cols_cv.index('y')], 
        fcsts_cv[:, cols_cv.index('SumAhead')]
    )
    fcsts_cv = fcsts_cv[:, cols_cv.index('SumAhead')].reshape((3, -1, h))
    actual_step_size = np.unique(
        np.diff(fcsts_cv, axis=1)
    )
    test_eq(actual_step_size, step_size)
    actual_n_windows = res_cv['forecasts'].shape[1]
    test_eq(actual_n_windows, int((test_size - h)/step_size) + 1)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
def fail_cv(h, test_size, step_size):
    return ga.cross_validation(models=[SumAhead()], h=h, test_size=test_size, step_size=step_size)
test_fail(fail_cv, contains='module', kwargs=dict(h=2, test_size=5, step_size=2))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#test fallback model
# cross validation
fcst_cv_f = ga.cross_validation(
    models=[NullModel(), NullModel()], 
    fallback_model=Naive(), h=2, 
    test_size=5,
    fitted=True
)
fcst_cv_naive = ga.cross_validation(
    models=[Naive(), Naive()], 
    h=2, 
    test_size=5,
    fitted=True
)
test_eq(fcst_cv_f['forecasts'], fcst_cv_naive['forecasts'])
np.testing.assert_array_equal(fcst_cv_f['fitted']['values'], fcst_cv_naive['fitted']['values'])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test cross validation without refit
res_cv_wo_refit = ga.cross_validation(models=[SumAhead()], h=2, test_size=5, refit=False, level=(50, 60))
res_cv_refit = ga.cross_validation(models=[SumAhead()], h=2, test_size=5, refit=True, level=(50, 60))
test_fail(test_eq, args=(res_cv_wo_refit['forecasts'], res_cv_refit['forecasts']))
#test first forecasts are equal
test_eq(
    res_cv_wo_refit['forecasts'][[0, 8, 16]],
    res_cv_refit['forecasts'][[0, 8, 16]]
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from statsforecast.models import AutoCES
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
res_cv_wo_refit = ga.cross_validation(models=[AutoCES()], h=2, test_size=5, refit=False, level=(50, 60))
res_cv_refit = ga.cross_validation(models=[AutoCES()], h=2, test_size=5, refit=True, level=(50, 60))
test_fail(test_eq, args=(res_cv_wo_refit['forecasts'], res_cv_refit['forecasts']))
#test first forecasts are equal
test_eq(
    res_cv_wo_refit['forecasts'][[0, 8, 16]],
    res_cv_refit['forecasts'][[0, 8, 16]]
)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
class DataFrameProcessing:
    """
    A utility to process Pandas or Polars dataframes for time series forecasting.

    This class ensures the dataframe is properly structured, with required columns
    ('unique_id', 'ds', 'y'), and the 'ds' column is of datetime type. It also
    provides options for sorting the dataframe based on a unique identifier and a
    timestamp, and separates the data into different arrays for easy access during
    forecasting operations.

    Attributes:
    ----------
    dataframe : pd.DataFrame or pl.DataFrame
        A pandas or polars dataframe to be processed.
    sort_dataframe : bool
        A boolean indicating whether the dataframe should be sorted.

    Methods:
    -------
    __call__():
        Processes the dataframe by ensuring the columns are in the correct format,
        sorts the dataframe if required, and separates the data into different
        arrays for future operations.
    _to_np_and_engine():
        Converts the dataframe to a numpy structured array and identifies the
        dataframe engine (pandas or polars).
    _validate_dataframe(dataframe: Union[pd.DataFrame, pl.DataFrame]):
        Checks if the required columns ('unique_id', 'ds', 'y') are present in the
        dataframe.
    _check_datetime(arr: np.array) -> np.array:
        Validates that the 'ds' column is of datetime type, and if not, attempts to
        convert it to datetime.
    """
    def __init__(
        self, 
        dataframe:Union[pd.DataFrame, pl.DataFrame],
        sort_dataframe:bool,
        validate:Optional[bool]=True,
        ):
        
        self.dataframe = dataframe
        self.sort_dataframe = sort_dataframe
        self.validate = validate
        
        # Columns declaration
        self.non_value_columns: Union[tuple, list] = ['unique_id', 'ds']
        self.datetime_column_name: str = 'ds'
        self.dt_dtype = np.dtype('datetime64')
        self.__call__()
    
    def __call__(self):
        """Sequential execution of the code"""
        # Declaring values that will be utilized 
        self.np_df = self._to_np_and_engine()
        self.dataframe_columns = self.np_df.dtype.names
        
        # Processing value columns 
        value_columns = [column for column in self.dataframe_columns if column not in self.non_value_columns]
        self.value_array = self.np_df[value_columns]
        if self.value_array.ndim == 1 and len(value_columns) > 1:
            self.value_array = (
                np.stack(
                    [
                        self.value_array[name].astype(float) 
                        for name in self.value_array.dtype.names]
                        ,axis=1
                    )
            )
        if self.value_array.ndim == 1 and len(value_columns) == 1:
            self.value_array = self.value_array[value_columns].astype(float).reshape(-1, 1)
        
        # Processing unique_id
        self.unique_id = self.np_df['unique_id']
        if self.unique_id.dtype.kind == 'O':
            self.unique_id.astype(str)

        # If values are already int or float then they won't be converted
        if self.unique_id.dtype.kind not in ['i', 'f']:
            # If all values in the numpy array are numerical then proceed with conversion
            if np.char.isnumeric(self.unique_id.astype(str)).all():
                # If number are whole then they will be converted to `int`, else `float`
                # This is pure aesthetics addition.
                self.unique_id = self.unique_id.astype(float)
                if np.isclose(self.unique_id, np.round(self.unique_id)).all():
                    self.unique_id = self.unique_id.astype(int)
        # NOTE: When sorting with Numpy, character values may be prioritized over numerical values if the data
        # type is set to 'object'. For instance, the value '10' would come before '3' because it contains '1' and '0'
        # at the beginning. One solution to this problem is to convert the data to 'float' if it is numerical.
        unique_id_count = pd.Series(self.unique_id).value_counts(sort=False)
        self.indices, sizes = unique_id_count.index, unique_id_count.values
        cum_sizes = np.cumsum(sizes)
        
        # Processing datestamp
        self.dates = self.np_df[self.datetime_column_name]
        if self.engine_dataframe==pd.DataFrame:
            self.dates = self.dataframe.index.get_level_values(self.datetime_column_name)
        self.dates = self.dates[cum_sizes - 1]
        self.indptr = np.append(0, cum_sizes).astype(np.int32)
        
        # Index that will be used by pandas, not polars
        self.index = pd.MultiIndex.from_arrays([self.np_df['unique_id'], self.np_df['ds'],], names=['unique_id', 'ds'])

    def grouped_array(self):
        return GroupedArray(self.value_array, self.indptr)
                
    def _to_np_and_engine(self):
        """
        This function will be utilised to convert DataFrame to dictionary.
        
        Returns:
            tuple[pd.DataFrame or pl.DataFrame, dict]: the engine that will be used to construct
                the output DataFrame and dictionary of DataFrame values
        
        Raises:
            ValueError: If DataFrame engine is not supported and/or accounted for.
        """

        ####################
        # Polars DataFrame #
        ####################
        if isinstance(self.dataframe, pl.DataFrame):
            # Ensure that all required columns are present in the DataFrame:
            self.engine_dataframe = pl.DataFrame
            if self.validate:
                self._validate_dataframe(self.dataframe)
            elif self.validate == False:
                self._partial_val_df(self.dataframe)

            # datetime check
            dt_arr = self.dataframe["ds"].to_numpy()
            processed_dt_arr = self._check_datetime(dt_arr)
            if type(dt_arr) != type(processed_dt_arr):
                self.dataframe = self.dataframe.with_columns(
                    pl.from_numpy(processed_dt_arr.to_numpy(), schema=["ds"])
                )

            sample_index_df = self.dataframe[self.non_value_columns]
            sorted_index_df = sample_index_df.sort(self.non_value_columns)
            is_monotonic_increasing = sample_index_df.frame_equal(sorted_index_df)

            # Sorting will be performed if sort is set to true and values are unsorted
            if not is_monotonic_increasing and self.sort_dataframe:
                self.dataframe = self.dataframe.sort(self.non_value_columns)

            # resources: https://github.com/pola-rs/polars/blob/4fca1ae51864f74e0367d8bc91b4a2db00e54174/py-polars/polars/dataframe/frame.py#L1975
            # resources: https://numpy.org/doc/stable/user/basics.rec.html
            # resources: https://numpy.org/doc/stable/reference/generated/numpy.core.records.fromarrays.html
            # NOTE: Structured array is not available in polars under the version 0.17.12
            pl_version = pkg_resources.get_distribution("polars").version
            min_pl_v = pkg_resources.parse_version("0.17.12")
            if pkg_resources.parse_version(pl_version) >= min_pl_v:
                return self.dataframe.to_numpy(structured=True)
            else:
                arrays = []
                for column, column_dtype in self.dataframe.schema.items():
                    ser = self.dataframe[column]
                    arr = ser.to_numpy()
                    arrays.append(
                        arr.astype(str, copy=False)
                        if str(column_dtype) == 'Utf8' and not ser.has_validity()
                        else arr
                    )
                arr_dtypes = list(zip(self.dataframe.columns, (a.dtype for a in arrays)))
                return np.rec.fromarrays(arrays, dtype=np.dtype(arr_dtypes))

        ####################
        # Pandas DataFrame #
        ####################
        elif isinstance(self.dataframe, pd.DataFrame):
            self.engine_dataframe = pd.DataFrame
            # Ensure that all required columns are present in the DataFrame:
            # Full validation
            if self.validate and self.dataframe.index.name == "unique_id":
                reset_df = self.dataframe.reset_index()
                self._validate_dataframe(reset_df)
                del reset_df
            
            elif self.validate and self.dataframe.index.name != "unique_id":
                self._validate_dataframe(self.dataframe)
                self.dataframe = self.dataframe.set_index('unique_id')
            
            # Partial validation
            elif self.validate == False and self.dataframe.index.name == "unique_id":
                reset_df = self.dataframe.reset_index()
                self._partial_val_df(reset_df)
                del reset_df
            
            elif self.validate == False and self.dataframe.index.name != "unique_id":
                self._partial_val_df(self.dataframe)
                self.dataframe = self.dataframe.set_index('unique_id')

            # Datetime check
            dt_arr = self.dataframe['ds'].values
            self.dataframe['ds'] = self._check_datetime(dt_arr)

            self.dataframe = self.dataframe.set_index('ds', append=True)
            
            # Sorting will be performed if sort is set to true and values are unsorted
            if not self.dataframe.index.is_monotonic_increasing and self.sort_dataframe:
                self.dataframe = self.dataframe.sort_values(self.non_value_columns)
 
            np_df = self.dataframe.to_records(index=True)

            return np_df
    
        ####################
        # Not Supported DF #
        ####################
        else:
            raise ValueError(f"{type(self.dataframe)} is not supported")

    def _validate_dataframe(self, dataframe:Union[pd.DataFrame, pl.DataFrame]):
        """
        Will ensure that all DataFrame columns match the required columns.

        This code requires a pandas DataFrame with the following structure:

        Columns:
        - `unique_id` Union[str, int, categorical]: an identifier for the series
        - `ds` Union[datestamp, int]: column should be either an integer indexing time or a
            datestamp ideally like YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp.
        - `y` Union[float, int]: represents the measurement we wish to forecast.

        Raise:
            KeyError: DataFrame is missing `unique_id`, `ds`, `y` columns.
        """
        required_columns = ['unique_id', 'ds', 'y']
        matches = all(rc in dataframe.columns for rc in required_columns)
        if not matches:
            raise KeyError("The DataFrame doesn't contain {} columns".format(", ".join(required_columns)))
    
    def _partial_val_df(self, dataframe:Union[pd.DataFrame, pl.DataFrame]):
        """
        Will ensure that all DataFrame columns match the required columns.

        This code requires a pandas DataFrame with the following structure:

        Columns:
        - `unique_id` Union[str, int, categorical]: an identifier for the series
        - `ds` Union[datestamp, int]: column should be either an integer indexing time or a
            datestamp ideally like YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp.
            
        Raise:
            KeyError: DataFrame is missing `unique_id` and/or `ds` columns.
        """
        required_columns = ['unique_id', 'ds']
        matches = all(rc in dataframe.columns for rc in required_columns)
        if not matches:
            raise KeyError("The DataFrame doesn't contain {} columns".format(", ".join(required_columns)))

    def _check_datetime(self, arr: np.ndarray) -> Union[pd.DatetimeIndex, np.ndarray]:
        dt_check = pd.api.types.is_datetime64_any_dtype(arr)
        int_float_check = arr.dtype.kind in ["i", "f"]
        if not dt_check and not int_float_check:
            self._ds_is_dt = True
            try:
                return pd.to_datetime(arr)
            except Exception as e:
                msg = (
                    "Failed to parse `ds` column as datetime. "
                    "Please use `pd.to_datetime` outside to fix the error. "
                    f"{e}"
                )
                raise Exception(msg) from e
        return arr
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
series = generate_series(10_000, n_static_features=2, equal_ends=False, engine='pandas')
sorted_series = series.sort_values(['unique_id', 'ds'])
unsorted_series = sorted_series.sample(frac=1.0)

df_process = DataFrameProcessing(dataframe=unsorted_series, sort_dataframe=True)
ga = df_process.grouped_array()
indices = df_process.indices
dates = df_process.dates
ds = df_process.index

np.testing.assert_allclose(ga.data, sorted_series.drop(columns='ds').values)
test_eq(indices, sorted_series.index.unique(level='unique_id'))
test_eq(dates, series.groupby('unique_id')['ds'].max().values)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
def test_gp_df(df, sort_df):
    df = df.set_index("ds", append=True)
    if not df.index.is_monotonic_increasing and sort_df:
        df = df.sort_index()
    data = df.values.astype(np.float32)
    indices_sizes = df.index.get_level_values("unique_id").value_counts(sort=False)
    indices = indices_sizes.index
    sizes = indices_sizes.values
    cum_sizes = sizes.cumsum()
    dates = df.index.get_level_values("ds")[cum_sizes - 1]
    indptr = np.append(0, cum_sizes).astype(np.int32)
    return GroupedArray(data, indptr), indices, dates, df.index
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def _cv_dates(last_dates, freq, h, test_size, step_size=1):
    #assuming step_size = 1
    if (test_size - h) % step_size:
        raise Exception('`test_size - h` should be module `step_size`')
    n_windows = int((test_size - h) / step_size) + 1
    if len(np.unique(last_dates)) == 1:
        if issubclass(last_dates.dtype.type, np.integer):
            total_dates = np.arange(last_dates[0] - test_size + 1, last_dates[0] + 1)
            out = np.empty((h * n_windows, 2), dtype=last_dates.dtype)
            freq = 1
        else:
            total_dates = pd.date_range(end=last_dates[0], periods=test_size, freq=freq)
            out = np.empty((h * n_windows, 2), dtype='datetime64[s]')
        for i_window, cutoff in enumerate(range(-test_size, -h + 1, step_size), start=0):
            end_cutoff = cutoff + h
            out[h * i_window : h * (i_window + 1), 0] = total_dates[cutoff:] if end_cutoff == 0 else total_dates[cutoff:end_cutoff]
            out[h * i_window : h * (i_window + 1), 1] = np.tile(total_dates[cutoff] - freq, h)
        dates = pd.DataFrame(np.tile(out, (len(last_dates), 1)), columns=['ds', 'cutoff'])
    else:
        dates = pd.concat([_cv_dates(np.array([ld]), freq, h, test_size, step_size) for ld in last_dates])
        dates = dates.reset_index(drop=True)
    return dates
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
ds_int_cv_test = pd.DataFrame({
    'ds': np.hstack([
        [46, 47, 48],
        [47, 48, 49],
        [48, 49, 50]
    ]),
    'cutoff': [45] * 3 + [46] * 3 + [47] * 3
}, dtype=np.int64)
test_eq(ds_int_cv_test, _cv_dates(np.array([50], dtype=np.int64), 'D', 3, 5))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
ds_int_cv_test = pd.DataFrame({
    'ds': np.hstack([
        [46, 47, 48],
        [48, 49, 50]
    ]),
    'cutoff': [45] * 3 + [47] * 3
}, dtype=np.int64)
test_eq(ds_int_cv_test, _cv_dates(np.array([50], dtype=np.int64), 'D', 3, 5, step_size=2))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
for e_e in [True, False]:
    n_series = 2

    df_process = DataFrameProcessing(dataframe=generate_series(n_series, equal_ends=e_e), sort_dataframe=True)
    ga = df_process.grouped_array()
    indices = df_process.indices
    dates = df_process.dates
    ds = df_process.index

    freq = pd.tseries.frequencies.to_offset('D')
    horizon = 3
    test_size = 5
    df_dates = _cv_dates(last_dates=dates, freq=freq, h=horizon, test_size=test_size)
    test_eq(len(df_dates), n_series * horizon * (test_size - horizon + 1)) 
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def _get_n_jobs(n_groups, n_jobs):
    if n_jobs == -1 or (n_jobs is None):
        actual_n_jobs = cpu_count()
    else:
        actual_n_jobs = n_jobs
    return min(n_groups, actual_n_jobs)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#tests for more series than resources
test_eq(_get_n_jobs(100, -1), cpu_count()) 
test_eq(_get_n_jobs(100, None), cpu_count())
test_eq(_get_n_jobs(100, 2), 2)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#tests for less series than resources
test_eq(_get_n_jobs(1, -1), 1) 
test_eq(_get_n_jobs(1, None), 1)
test_eq(_get_n_jobs(2, 10), 2)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def _parse_ds_type(df):
    dt_col = df['ds']
    dt_check = pd.api.types.is_datetime64_any_dtype(dt_col)
    int_float_check = dt_col.dtype.kind in ['i', 'f']
    if not dt_check and not int_float_check:
        df = df.copy()
        try:
            df['ds'] = pd.to_datetime(df['ds'])
        except Exception as e:
            msg = (
                'Failed to parse `ds` column as datetime. '
                'Please use `pd.to_datetime` outside to fix the error. '
                f'{e}'
            )
            raise Exception(msg) from e
    return df
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
class _StatsForecast:
    
    def __init__(
            self, 
            models: List[Any],
            freq: str,
            n_jobs: int = 1,
            df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            sort_df: bool = True,
            fallback_model: Optional[Any] = None,
            verbose: bool = False,
        ):
        """Train statistical models.

        The `StatsForecast` class allows you to efficiently fit multiple `StatsForecast` models 
        for large sets of time series. It operates with pandas DataFrame `df` that identifies series 
        and datestamps with the `unique_id` and `ds` columns. The `y` column denotes the target 
        time series variable. 

        The class has memory-efficient `StatsForecast.forecast` method that avoids storing partial 
        model outputs. While the `StatsForecast.fit` and `StatsForecast.predict` methods with 
        Scikit-learn interface store the fitted models.

        The `StatsForecast` class offers parallelization utilities with Dask, Spark and Ray back-ends.
        See distributed computing example [here](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray).

        Parameters
        ----------
        models : List[Any]
            List of instantiated objects models.StatsForecast.
        freq : str
            Frequency of the data.
            See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
        n_jobs : int (default=1)
            Number of jobs used in the parallel processing, use -1 for all cores.
        df : pandas.DataFrame or pl.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous.
        sort_df : bool (default=True)
            If True, sort `df` by [`unique_id`,`ds`].
        fallback_model : Any, optional (default=None)
            Model to be used if a model fails. 
            Only works with the `forecast` and `cross_validation` methods.
        verbose : bool (default=True)
            Prints TQDM progress bar when `n_jobs=1`.
        """
    
        # TODO @fede: needed for residuals, think about it later
        self.models = models
        self._validate_model_names()
        self.freq = pd.tseries.frequencies.to_offset(freq)
        self.n_jobs = n_jobs
        self.fallback_model = fallback_model
        self.verbose = verbose 
        self.n_jobs == 1
        self._prepare_fit(df=df, sort_df=sort_df)

    def _validate_model_names(self):
        # Some test models don't have alias
        names = [getattr(model, 'alias', lambda: None) for model in self.models]
        names = [x for x in names if x is not None]
        if len(names) != len(set(names)):
            raise ValueError('Model names must be unique. You can use `alias` to set a unique name for each model.')

    def _prepare_fit(self, df, sort_df):
        if df is not None:
            df_process = DataFrameProcessing(df, sort_df)
            self.ga = df_process.grouped_array()
            self.uids = df_process.indices
            self.last_dates = df_process.dates
            self.ds = df_process.index
            self.og_dates = df_process.np_df['ds']
            self.og_unique_id = df_process.np_df['unique_id']
            self.engine = df_process.engine_dataframe
            self.n_jobs = _get_n_jobs(len(self.ga), self.n_jobs)
            self.sort_df = sort_df
            
    def _set_prediction_intervals(self, prediction_intervals):
        for model in self.models:
            interval = getattr(model, "prediction_intervals", None)
            if interval is None:
                setattr(model, "prediction_intervals", prediction_intervals)
        
    def fit(
            self,
            df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None, 
            sort_df: bool = True,
            prediction_intervals: Optional[ConformalIntervals] = None,
        ):
        """Fit statistical models.

        Fit `models` to a large set of time series from DataFrame `df`
        and store fitted models for later inspection.

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous.
            If None, the `StatsForecast` class should have been instantiated
            using `df`.
        sort_df : bool (default=True)
            If True, sort `df` by [`unique_id`,`ds`].
        prediction_intervals : ConformalIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).

        Returns
        -------
        self : StatsForecast
            Returns with stored `StatsForecast` fitted `models`.
        """
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        self._prepare_fit(df, sort_df)
        if self.n_jobs == 1:
            self.fitted_ = self.ga.fit(models=self.models)
        else:
            self.fitted_ = self._fit_parallel()
        return self
    
    def _make_future_df(self, h: int):
        if issubclass(self.last_dates.dtype.type, np.integer):
            last_date_f = lambda x: np.arange(x + 1, x + 1 + h, dtype=self.last_dates.dtype)
        else:
            last_date_f = lambda x: pd.date_range(x + self.freq, periods=h, freq=self.freq)
        if len(np.unique(self.last_dates)) == 1:
            dates = np.tile(last_date_f(self.last_dates[0]), len(self.ga))
        else:
            dates = np.hstack([
                last_date_f(last_date)
                for last_date in self.last_dates            
            ])
        u_id_ser:Union[pd.Series, pl.Series] = np.repeat(self.uids, h)
        unique_id: np.ndarray = u_id_ser.to_numpy()

        # In older versions to_numpy converts string values into object,
        # creating bytes error, this fixes it
        if unique_id.dtype.kind == 'O':
            unique_id = unique_id.astype(str)

        if self.engine == pd.DataFrame:
            idx = pd.Index(unique_id, name='unique_id')
            df = self.engine({'ds': dates}, index=idx)
        elif self.engine == pl.DataFrame:
            df = self.engine({'unique_id': unique_id, 'ds': dates})
        return df
    
    def _parse_X_level(self, h, X, level):
        if X is not None:
            if isinstance(X, pd.DataFrame):
                if X.index.name != "unique_id":
                    X = X.set_index("unique_id")
            expected_shape_rows = h * len(self.ga)
            ga_shape = self.ga.data.shape[1]
            # Polars doesn't have index, hence, extra "column"
            expected_shape_cols = ga_shape if not isinstance(X, pl.DataFrame) else ga_shape+1
            expected_shape = (expected_shape_rows, expected_shape_cols)

            if X.shape != expected_shape:
                raise ValueError(f'Expected X to have shape {expected_shape}, but got {X.shape}')
            X = DataFrameProcessing(X, sort_dataframe=self.sort_df, validate=False).grouped_array()
        if level is None:
            level = tuple()
        return X, level
    
    def predict(
            self,
            h: int,
            X_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            level: Optional[List[int]] = None,
        ):
        """Predict statistical models.

        Use stored fitted `models` to predict large set of time series from DataFrame `df`.        

        Parameters
        ----------
        h : int
            Forecast horizon.
        X_df : pandas.DataFrame | polars.DataFrame, optional (default=None)
            DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
        level : List[float], optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.

        Returns
        -------
        fcsts_df : pandas.DataFrame | polars.DataFrame
            DataFrame with `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        X, level = self._parse_X_level(h=h, X=X_df, level=level)
        if self.n_jobs == 1:
            fcsts, cols = self.ga.predict(fm=self.fitted_, h=h, X=X, level=level)
        else:
            fcsts, cols = self._predict_parallel(h=h, X=X, level=level)
        fcsts_df = self._make_future_df(h=h)
        fcsts_df[cols] = fcsts
        return fcsts_df
    
    def fit_predict(
            self,
            h: int,
            df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            X_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            level: Optional[List[int]] = None,
            sort_df: bool = True,
            prediction_intervals: Optional[ConformalIntervals] = None,
        ):
        """Fit and Predict with statistical models.

        This method avoids memory burden due from object storage.
        It is analogous to Scikit-Learn `fit_predict` without storing information.
        It requires the forecast horizon `h` in advance. 
        
        In contrast to `StatsForecast.forecast` this method stores partial models outputs.

        Parameters
        ----------
        h : int
            Forecast horizon.
        df : pandas.DataFrame | polars.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If None, the `StatsForecast` class should have been instantiated
            using `df`.
        X_df : pandas.DataFrame | polars.DataFrame, optional (default=None)
            DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
        level : List[float], optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        sort_df : bool (default=True)
            If True, sort `df` by [`unique_id`,`ds`].
        prediction_intervals : ConformalIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).

        Returns
        -------
        fcsts_df : pandas.DataFrame | polars.DataFrame
            DataFrame with `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        self._prepare_fit(df, sort_df)
        X, level = self._parse_X_level(h=h, X=X_df, level=level)
        if self.n_jobs == 1:
            self.fitted_, fcsts, cols = self.ga.fit_predict(models=self.models, h=h, X=X, level=level)
        else:
            self.fitted_, fcsts, cols = self._fit_predict_parallel(h=h, X=X, level=level)
        fcsts_df = self._make_future_df(h=h)
        fcsts_df[cols] = fcsts
        return fcsts_df
    
    def forecast(
            self,
            h: int,
            df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            X_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            level: Optional[List[int]] = None,
            fitted: bool = False,
            sort_df: bool = True,
            prediction_intervals: Optional[ConformalIntervals] = None,
        ):
        """Memory Efficient predictions.

        This method avoids memory burden due from object storage.
        It is analogous to Scikit-Learn `fit_predict` without storing information.
        It requires the forecast horizon `h` in advance.

        Parameters
        ----------
        h : int
            Forecast horizon.
        df : pandas.DataFrame | polars.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous.
            If None, the `StatsForecast` class should have been instantiated
            using `df`.
        X_df : pandas.DataFrame | polars.DataFrame, optional (default=None)
            DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
        level : List[float], optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        fitted : bool (default=False)
            Wether or not return insample predictions.
        sort_df : bool (default=True)
            If True, sort `df` by [`unique_id`,`ds`].
        prediction_intervals : ConformalIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).
        
        Returns
        -------
        fcsts_df : pandas.DataFrame | polars.DataFrame
            DataFrame with `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        self._prepare_fit(df, sort_df)
        X, level = self._parse_X_level(h=h, X=X_df, level=level)
        if self.n_jobs == 1:
            res_fcsts = self.ga.forecast(models=self.models, 
                                         h=h, fallback_model=self.fallback_model, 
                                         fitted=fitted, X=X, level=level, 
                                         verbose=self.verbose)
        else:
            res_fcsts = self._forecast_parallel(h=h, fitted=fitted, X=X, level=level)
        if fitted:
            self.fcst_fitted_values_ = res_fcsts['fitted']
        fcsts = res_fcsts['forecasts']
        cols = res_fcsts['cols']
        fcsts_df = self._make_future_df(h=h)
        fcsts_df[cols] = fcsts
        return fcsts_df
    
    def forecast_fitted_values(self):
        """Access insample predictions.

        After executing `StatsForecast.forecast`, you can access the insample 
        prediction values for each model. To get them, you need to pass `fitted=True` 
        to the `StatsForecast.forecast` method and then use the 
        `StatsForecast.forecast_fitted_values` method.
        
        Parameters
        ----------
        self : StatsForecast

        Returns
        -------
        fcsts_df : pandas.DataFrame | polars.DataFrame
            DataFrame with insample `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        if not hasattr(self, "fcst_fitted_values_"):
            raise Exception("Please run `forecast` mehtod using `fitted=True`")
        cols = self.fcst_fitted_values_["cols"]
        if self.engine == pd.DataFrame:
            df = self.engine(
                self.fcst_fitted_values_["values"], columns=cols, index=self.ds
            ).reset_index(level=1)
        elif self.engine == pl.DataFrame:
            df = self.engine({'unique_id': self.og_unique_id, 'ds': self.og_dates})
            df[cols] = self.fcst_fitted_values_["values"]
        return df
    
    def cross_validation(
            self,
            h: int,
            df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            n_windows: int = 1,
            step_size: int = 1,
            test_size: Optional[int] = None,
            input_size: Optional[int] = None,
            level: Optional[List[int]] = None,
            fitted: bool = False,
            refit: bool = True,
            sort_df: bool = True,
            prediction_intervals: Optional[ConformalIntervals] = None,
        ):
        """Temporal Cross-Validation.

        Efficiently fits a list of `StatsForecast` 
        models through multiple training windows, in either chained or rolled manner.
        
        `StatsForecast.models`' speed allows to overcome this evaluation technique 
        high computational costs. Temporal cross-validation provides better model's 
        generalization measurements by increasing the test's length and diversity.

        Parameters
        ----------
        h : int 
            Forecast horizon.
        df : pandas.DataFrame | polars.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous.
            If None, the `StatsForecast` class should have been instantiated
            using `df`.
        n_windows : int (default=1)
            Number of windows used for cross validation.
        step_size : int (default=1)
            Step size between each window.
        test_size : int, optional (default=None)
            Length of test size. If passed, set `n_windows=None`.
        input_size : int, optional (default=None)
            Input size for each window, if not none rolled windows.
        level : List[float], optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        fitted : bool (default=False)
            Wether or not returns insample predictions.
        refit : bool (default=True)
            Wether or not refit the model for each window.
        sort_df : bool (default=True)
            If True, sort `df` by `unique_id` and `ds`.
        prediction_intervals : ConformalIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).

        Returns
        -------
        fcsts_df : pandas.DataFrame
            DataFrame with insample `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        if test_size is None:
            test_size = h + step_size * (n_windows - 1)
        elif n_windows is None:
            if (test_size - h) % step_size:
                raise Exception('`test_size - h` should be module `step_size`')
            n_windows = int((test_size - h) / step_size) + 1
        elif (n_windows is None) and (test_size is None):
            raise Exception('you must define `n_windows` or `test_size`')
        else:
            raise Exception('you must define `n_windows` or `test_size` but not both')
        self._set_prediction_intervals(prediction_intervals=prediction_intervals)
        self._prepare_fit(df, sort_df)
        _, level = self._parse_X_level(h=h, X=None, level=level)
        if self.n_jobs == 1:
            res_fcsts = self.ga.cross_validation(
                models=self.models, h=h, test_size=test_size, 
                fallback_model=self.fallback_model, 
                step_size=step_size, 
                input_size=input_size, 
                fitted=fitted,
                level=level,
                verbose=self.verbose,
                refit=refit
            )
        else:
            res_fcsts = self._cross_validation_parallel(
                h=h, 
                test_size=test_size,
                step_size=step_size,
                input_size=input_size,
                fitted=fitted,
                level=level,
                refit=refit
            )
            
        if fitted:
            self.cv_fitted_values_ = res_fcsts['fitted']
            self.n_cv_ = n_windows
            
        fcsts = res_fcsts['forecasts']
        cols = res_fcsts['cols']
        fcsts_df = _cv_dates(last_dates=self.last_dates, freq=self.freq, 
                             h=h, test_size=test_size, step_size=step_size)
        idx = pd.Index(np.repeat(self.uids, h * n_windows), name='unique_id')
        fcsts_df.index = idx
        fcsts_df[cols] = fcsts
        if self.engine == pl.DataFrame:
            fcsts_df = pl.from_pandas(fcsts_df, include_index=True)
        return fcsts_df
    
    def cross_validation_fitted_values(self):
        """Access insample cross validated predictions.

        After executing `StatsForecast.cross_validation`, you can access the insample 
        prediction values for each model and window. To get them, you need to pass `fitted=True` 
        to the `StatsForecast.cross_validation` method and then use the 
        `StatsForecast.cross_validation_fitted_values` method.
        
        Parameters
        ----------
        self : StatsForecast

        Returns
        -------
        fcsts_df : pandas.DataFrame | polars.DataFrame
            DataFrame with insample `models` columns for point predictions 
            and probabilistic predictions for all fitted `models`.
        """
        if not hasattr(self, 'cv_fitted_values_'):
            raise Exception('Please run `cross_validation` mehtod using `fitted=True`')
        index = pd.MultiIndex.from_tuples(np.tile(self.ds, self.n_cv_), names=['unique_id', 'ds'])
        df = pd.DataFrame(index=index)
        df['cutoff'] = self.cv_fitted_values_['last_idxs'].flatten(order='F')
        df[self.cv_fitted_values_['cols']] = np.reshape(self.cv_fitted_values_['values'], (-1, len(self.models) + 1), order='F')
        idxs = self.cv_fitted_values_['idxs'].flatten(order='F')
        df = df.iloc[idxs].reset_index(level=1)
        df['cutoff'] = df['ds'].where(df['cutoff']).bfill()

        if self.engine == pl.DataFrame:
            df = pl.from_pandas(df, include_index=True)
        return df

    def _get_pool(self):
        from multiprocessing import Pool

        pool_kwargs = dict()
        return Pool, pool_kwargs
    
    def _fit_parallel(self):
        gas = self.ga.split(self.n_jobs)
        Pool, pool_kwargs = self._get_pool()
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga in gas:
                future = executor.apply_async(ga.fit, (self.models,))
                futures.append(future)
            fm = np.vstack([f.get() for f in futures])
        return fm
    
    def _get_gas_Xs(self, X):
        gas = self.ga.split(self.n_jobs)
        if X is not None:
            Xs = X.split(self.n_jobs)
        else:
            from itertools import repeat
            Xs = repeat(None)
        return gas, Xs
    
    def _predict_parallel(self, h, X, level):
        #create elements for each core
        gas, Xs = self._get_gas_Xs(X=X)
        fms = self.ga.split_fm(self.fitted_, self.n_jobs)
        Pool, pool_kwargs = self._get_pool()
        #compute parallel forecasts
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga, fm, X_ in zip(gas, fms, Xs):
                future = executor.apply_async(ga.predict, (fm, h, X_, level,))
                futures.append(future)
            out = [f.get() for f in futures]
            fcsts, cols = list(zip(*out))
            fcsts = np.vstack(fcsts)
            cols = cols[0]
        return fcsts, cols
    
    def _fit_predict_parallel(self, h, X, level):
        #create elements for each core
        gas, Xs = self._get_gas_Xs(X=X)
        Pool, pool_kwargs = self._get_pool()
        #compute parallel forecasts
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga, X_ in zip(gas, Xs):
                future = executor.apply_async(ga.fit_predict, (self.models, h, X_, level,))
                futures.append(future)
            out = [f.get() for f in futures]
            fm, fcsts, cols = list(zip(*out))
            fm = np.vstack(fm)
            fcsts = np.vstack(fcsts)
            cols = cols[0]
        return fm, fcsts, cols
    
    def _forecast_parallel(self, h, fitted, X, level):
        #create elements for each core
        gas, Xs = self._get_gas_Xs(X=X)
        Pool, pool_kwargs = self._get_pool()
        #compute parallel forecasts
        result = {}
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga, X_ in zip(gas, Xs):
                future = executor.apply_async(
                    ga.forecast, 
                    (self.models, h, self.fallback_model, fitted, X_, level,)
                )
                futures.append(future)
            out = [f.get() for f in futures]
            fcsts = [d['forecasts'] for d in out]
            fcsts = np.vstack(fcsts)
            cols = out[0]['cols']
            result['forecasts'] = fcsts
            result['cols'] = cols
            if fitted:
                result['fitted'] = {}
                fitted_vals = [d['fitted']['values'] for d in out]
                result['fitted']['values'] = np.vstack(fitted_vals)
                result['fitted']['cols'] = out[0]['fitted']['cols']
        return result
    
    def _cross_validation_parallel(self, h, test_size, step_size, input_size, fitted, level, refit):
        #create elements for each core
        gas = self.ga.split(self.n_jobs)
        Pool, pool_kwargs = self._get_pool()
        #compute parallel forecasts
        result = {}
        with Pool(self.n_jobs, **pool_kwargs) as executor:
            futures = []
            for ga in gas:
                future = executor.apply_async(
                    ga.cross_validation, 
                    (self.models, h, test_size, self.fallback_model, step_size, input_size, fitted, level, refit,)
                )
                futures.append(future)
            out = [f.get() for f in futures]
            fcsts = [d['forecasts'] for d in out]
            fcsts = np.vstack(fcsts)
            cols = out[0]['cols']
            result['forecasts'] = fcsts
            result['cols'] = cols
            if fitted:
                result['fitted'] = {}
                result['fitted']['values'] = np.concatenate([d['fitted']['values'] for d in out])
                for key in ['last_idxs', 'idxs']:
                    result['fitted'][key] = np.concatenate([d['fitted'][key] for d in out])
                result['fitted']['cols'] = out[0]['fitted']['cols']
        return result
    
    @staticmethod
    def plot(df: Union[pd.DataFrame, pl.DataFrame],
             forecasts_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
             unique_ids: Union[Optional[List[str]], np.ndarray] = None,
             plot_random: bool = True, 
             models: Optional[List[str]] = None, 
             level: Optional[List[float]] = None,
             max_insample_length: Optional[int] = None,
             plot_anomalies: bool = False,
             engine: str = 'plotly',
             resampler_kwargs: Optional[Dict] = None):
        """Plot forecasts and insample values.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns [`unique_id`, `ds`, `y`].
        forecasts_df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`] and models.
        unique_ids : List[str], optional (default=None)
            Time Series to plot.
            If None, time series are selected randomly.
        plot_random : bool (default=True)
            Select time series to plot randomly.
        models : List[str], optional (default=None)
            List of models to plot.
        level : List[float], optional (default=None)
            List of prediction intervals to plot if paseed.
        max_insample_length : int, optional (default=None)
            Max number of train/insample observations to be plotted.
        plot_anomalies : bool (default=False)
            Plot anomalies for each prediction interval.
        engine : str (default='plotly')
            Library used to plot. 'plotly', 'plotly-resampler' or 'matplotlib'.
        resampler_kwargs : dict
            Kwargs to be passed to plotly-resampler constructor. 
            For further custumization ("show_dash") call the method,
            store the plotting object and add the extra arguments to
            its `show_dash` method.
        """

        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        if isinstance(forecasts_df, pl.DataFrame):
            forecasts_df = forecasts_df.to_pandas()

        if level is not None and not isinstance(level, list):
            raise Exception(
                'Please use a list for the `level` argument '
                'If you only have one level, use `level=[your_level]`'
            )

        if unique_ids is None:
            df_pt = DataFrameProcessing(dataframe=df, sort_dataframe=True, validate=False)
            uids_arr: pd.Index = df_pt.indices
            uid_dtype = uids_arr.dtype

            if df.index.name != 'unique_id':
                df['unique_id'] = df['unique_id'].astype(uid_dtype)
                df = df.set_index('unique_id')
            else:
                df.index = df.index.astype(uid_dtype)

            if forecasts_df is not None:
                if isinstance(forecasts_df, pl.DataFrame):
                    forecasts_df = forecasts_df.to_pandas()

                if forecasts_df.index.name == 'unique_id':
                    forecasts_df.index = forecasts_df.index.astype(uid_dtype)
                    unique_ids = np.intersect1d(uids_arr, forecasts_df.index.unique())
                else:
                    forecasts_df['unique_id'] = forecasts_df['unique_id'].astype(uid_dtype)
                    unique_ids = np.intersect1d(uids_arr, forecasts_df['unique_id'].unique())
            else:
                unique_ids = uids_arr

        if plot_random:
            unique_ids = random.sample(list(unique_ids), k=min(8, len(unique_ids)))
        else:
            unique_ids = unique_ids[:8]

        if engine in ["plotly", "plotly-resampler"]:
            n_rows = min(4, len(unique_ids) // 2 + 1 if len(unique_ids) > 2 else 1)
            fig = make_subplots(rows=n_rows, cols=2 if len(unique_ids) >= 2 else 1, 
                                vertical_spacing=0.1, 
                                horizontal_spacing=0.07, 
                                x_title='Datestamp [ds]',
                                y_title='Target [y]',
                                subplot_titles=[str(uid) for uid in unique_ids])
            if engine == "plotly-resampler":
                try:
                    from plotly_resampler import FigureResampler
                except ImportError:
                    raise ImportError(
                        "plotly-resampler is not installed. "
                        "Please install it with `pip install plotly-resampler`"
                    )
                resampler_kwargs = {} if resampler_kwargs is None else resampler_kwargs
                fig = FigureResampler(fig, **resampler_kwargs)
            showed_legends: set = set()
            
            def plotly(df, fig, n_rows, unique_ids, models, 
                       plot_anomalies, max_insample_length,
                       showed_legends):
                if models is None:
                    exclude_str = ['lo', 'hi', 'unique_id', 'ds']
                    models = [c for c in df.columns if all(item not in c for item in exclude_str)]
                if 'y' not in models:
                    models = ['y'] + models
                for uid, (idx, idy) in zip(unique_ids, product(range(1, n_rows + 1), range(1, 2 + 1))):
                    df_uid = df.query('unique_id == @uid')
                    if max_insample_length:
                        df_uid = df_uid.iloc[-max_insample_length:]
                    plot_anomalies = 'y' in df_uid and plot_anomalies
                    df_uid = _parse_ds_type(df_uid)
                    colors = plt.cm.get_cmap('tab20b', len(models))
                    colors = ['#1f77b4'] + [cm.to_hex(colors(i)) for i in range(len(models))]
                    for col, color in zip(models, colors):
                        if col in df_uid:
                            model = df_uid[col]
                            fig.add_trace(
                                go.Scatter(x=df_uid['ds'], 
                                           y=model, 
                                           mode='lines', 
                                           name=col, 
                                           legendgroup=col,
                                           line=dict(color=color, width=1), 
                                           showlegend=(idx==1 and idy==1 and col not in showed_legends)),
                                row=idx, col=idy
                            )
                            showed_legends.add(col)
                        model_has_level = any(f'{col}-lo' in c for c in df_uid)
                        if level is not None and model_has_level:
                            level_ = level
                        elif model_has_level:
                            level_col = df_uid.filter(like=f'{col}-lo').columns[0]
                            level_col = re.findall('[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+', level_col)[0]
                            level_ = [level_col]
                        else:
                            level_ = []
                        ds = df_uid['ds']    
                        for lv in level_:
                            lo = df_uid[f'{col}-lo-{lv}']
                            hi = df_uid[f'{col}-hi-{lv}']
                            plot_name = f'{col}_level_{lv}'
                            fig.add_trace(
                                go.Scatter(x=np.concatenate([ds, ds[::-1]]), 
                                           y=np.concatenate([hi, lo[::-1]]), 
                                           fill='toself',
                                           mode='lines',
                                           fillcolor=color,
                                           opacity=-float(lv)/100 + 1,
                                           name=plot_name, 
                                           legendgroup=plot_name,
                                           line=dict(color=color, width=1), 
                                           showlegend=(idx==1 and idy==1 and plot_name not in showed_legends)),
                                row=idx, col=idy
                            )
                            showed_legends.add(plot_name)
                            if col != 'y' and plot_anomalies:
                                anomalies = (df_uid['y'] < lo) | (df_uid['y'] > hi)
                                plot_name = f'{col}_anomalies_level_{lv}'
                                fig.add_trace(
                                    go.Scatter(x=ds[anomalies], 
                                               y=df_uid['y'][anomalies], 
                                               fillcolor=color,
                                               mode='markers',
                                               opacity=float(lv)/100,
                                               name=plot_name, 
                                               legendgroup=plot_name,
                                               line=dict(color=color, width=0.7), 
                                               marker=dict(size=4, line=dict(color='red', width=0.5)),
                                               showlegend=(idx==1 and idy==1 and plot_name not in showed_legends)),
                                    row=idx, col=idy
                                )
                                showed_legends.add(plot_name)
                return fig
            fig = plotly(df=df, fig=fig, n_rows=n_rows, 
                         unique_ids=unique_ids, 
                         models=models, 
                         plot_anomalies=plot_anomalies, 
                         max_insample_length=max_insample_length,
                         showed_legends=showed_legends)
            if forecasts_df is not None:
                fig = plotly(df=forecasts_df, 
                             fig=fig, n_rows=n_rows, 
                             unique_ids=unique_ids, 
                             models=models, 
                             plot_anomalies=plot_anomalies, 
                             max_insample_length=None,
                             showed_legends=showed_legends)
            fig.update_xaxes(matches=None, showticklabels=True, visible=True)
            fig.update_layout(margin=dict(l=60, r=10, t=20, b=50))
            fig.update_layout(template="plotly_white", font=dict(size=10))
            fig.update_annotations(font_size=10)
            fig.update_layout(autosize=True, height=150 * n_rows)
            
        elif engine == 'matplotlib':
            if len(unique_ids) == 1:
                fig, axes = plt.subplots(figsize = (24, 3.5))
                axes = np.array([[axes]])
                n_cols = 1
            else:
                n_cols = min(4, len(unique_ids) // 2 + 1 if len(unique_ids) > 2 else 1)
                fig, axes = plt.subplots(n_cols, 2, figsize = (24, 3.5 * n_cols))
                if n_cols == 1:
                    axes = np.array([axes])

            for uid, (idx, idy) in zip(unique_ids, product(range(n_cols), range(2))):
                train_uid = df.query('unique_id == @uid')
                train_uid = _parse_ds_type(train_uid)
                if max_insample_length is not None:
                    train_uid = train_uid.iloc[-max_insample_length:]
                ds = train_uid['ds']
                y = train_uid['y']
                axes[idx, idy].plot(ds, y, label = 'y')
                if forecasts_df is not None:
                    if models is None:
                        exclude_str = ['lo', 'hi', 'unique_id', 'ds']
                        models = [c for c in forecasts_df.columns if all(item not in c for item in exclude_str)]
                    if 'y' not in models:
                        models = ['y'] + models
                    test_uid = forecasts_df.query('unique_id == @uid')
                    plot_anomalies = 'y' in test_uid and plot_anomalies
                    test_uid = _parse_ds_type(test_uid)
                    first_ds_fcst = test_uid['ds'].min()
                    axes[idx, idy].axvline(x=first_ds_fcst, 
                                           color='black', 
                                           label='First ds Forecast', 
                                           linestyle='--')
                    colors = plt.cm.get_cmap('tab20b', len(models))
                    colors = ['blue'] + [colors(i) for i in range(len(models))]
                    for col, color in zip(models, colors):
                        if col in test_uid:
                            axes[idx, idy].plot(test_uid['ds'], test_uid[col], label=col, color=color)
                        model_has_level = any(f'{col}-lo' in c for c in test_uid)
                        if level is not None and model_has_level:
                            level_ = level
                        elif model_has_level:
                            level_col = test_uid.filter(like=f'{col}-lo').columns[0]
                            level_col = re.findall('[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+', level_col)[0]
                            level_ = [level_col]
                        else:
                            level_ = []
                        for lv in level_:
                            ds_test = test_uid['ds']
                            lo = test_uid[f'{col}-lo-{lv}']
                            hi = test_uid[f'{col}-hi-{lv}']
                            axes[idx, idy].fill_between(
                                ds_test, 
                                lo, 
                                hi,
                                alpha=-float(lv)/100 + 1,
                                color=color,
                                label=f'{col}_level_{lv}',
                            )
                            if col != 'y' and plot_anomalies:
                                anomalies = (test_uid['y'] < lo) | (test_uid['y'] > hi)
                                axes[idx, idy].scatter(
                                    x=ds_test[anomalies], 
                                    y=test_uid['y'][anomalies], 
                                    color=color,
                                    s=30,
                                    alpha=float(lv)/100,
                                    label=f'{col}_anomalies_level_{lv}', 
                                    linewidths=0.5,
                                    edgecolors='red',
                                )

                axes[idx, idy].set_title(f'{uid}')
                axes[idx, idy].set_xlabel('Datestamp [ds]')
                axes[idx, idy].set_ylabel('Target [y]')
                axes[idx, idy].legend(loc='upper left')
                axes[idx, idy].xaxis.set_major_locator(plt.MaxNLocator(min(len(df) // 30, 10)))
                axes[idx, idy].grid()
            fig.subplots_adjust(hspace=0.5)
            plt.close(fig)
        else:
            raise Exception(f'Unkwon plot engine {engine}')
        return fig
    
    def __repr__(self):
        return f"StatsForecast(models=[{','.join(map(repr, self.models))}])"
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
class ParallelBackend:
    def forecast(self, df, models, freq, fallback_model=None, **kwargs: Any) -> Any:
        model = _StatsForecast(df=df, models=models, freq=freq, fallback_model=fallback_model)
        return model.forecast(**kwargs)

    def cross_validation(self, df, models, freq, fallback_model=None, **kwargs: Any) -> Any:
        model = _StatsForecast(df=df, models=models, freq=freq, fallback_model=fallback_model)
        return model.cross_validation(**kwargs)
    

@conditional_dispatcher
def make_backend(obj:Any, *args:Any, **kwargs:Any) -> ParallelBackend:
    return ParallelBackend()
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class StatsForecast(_StatsForecast):
    """Train statistical models.

    The `StatsForecast` class allows you to efficiently fit multiple `StatsForecast` models 
    for large sets of time series. It operates with pandas DataFrame `df` that identifies series 
    and datestamps with the `unique_id` and `ds` columns. The `y` column denotes the target 
    time series variable. 

    The class has memory-efficient `StatsForecast.forecast` method that avoids storing partial 
    model outputs. While the `StatsForecast.fit` and `StatsForecast.predict` methods with 
    Scikit-learn interface store the fitted models.

    The `StatsForecast` class offers parallelization utilities with Dask, Spark and Ray back-ends.
    See distributed computing example [here](https://github.com/Nixtla/statsforecast/tree/main/experiments/ray).

    Parameters
    ----------
    models : List[Any]
        List of instantiated objects models.StatsForecast.
    freq : str
        Frequency of the data.
        See [panda's available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
    n_jobs : int (default=1)
        Number of jobs used in the parallel processing, use -1 for all cores.
    df : pandas.DataFrame | pl.DataFrame, optional (default=None)
        DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous.
    sort_df : bool (default=True)
        If True, sort `df` by [`unique_id`,`ds`].
    fallback_model : Any, optional (default=None)
        Model to be used if a model fails. 
        Only works with the `forecast` and `cross_validation` methods.
    verbose : bool (default=True)
        Prints TQDM progress bar when `n_jobs=1`.
    """

    def forecast(
            self,
            h: int,
            df: Any = None,
            X_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            level: Optional[List[int]] = None,
            fitted: bool = False,
            sort_df: bool = True,
            prediction_intervals: Optional[ConformalIntervals] = None,
        ):
        if self._is_native(df=df):
            return super().forecast(
                h=h,
                df=df,
                X_df=X_df,
                level=level,
                fitted=fitted,
                sort_df=sort_df,
                prediction_intervals=prediction_intervals,
            )
        assert df is not None
        engine = make_execution_engine(infer_by=[df])
        backend = make_backend(engine)
        return backend.forecast(
            df=df,
            models=self.models,
            freq=self.freq,
            fallback_model=self.fallback_model,
            h=h,
            X_df=X_df,
            level=level,
            fitted=fitted,
            prediction_intervals=prediction_intervals,
        )
    
    def cross_validation(
            self,
            h: int,
            df: Any = None,
            n_windows: int = 1,
            step_size: int = 1,
            test_size: Optional[int] = None,
            input_size: Optional[int] = None,
            level: Optional[List[int]] = None,
            fitted: bool = False,
            refit: bool = True,
            sort_df: bool = True,
            prediction_intervals: Optional[ConformalIntervals] = None,
        ):
        if self._is_native(df=df):
            return super().cross_validation(
                h=h,
                df=df,
                n_windows=n_windows,
                step_size=step_size,
                test_size=test_size,
                input_size=input_size,
                level=level,
                fitted=fitted,
                refit=refit,
                sort_df=sort_df,
                prediction_intervals=prediction_intervals,
            )
        assert df is not None
        engine = make_execution_engine(infer_by=[df])
        backend = make_backend(engine)
        return backend.cross_validation(
            df=df,
            models=self.models,
            freq=self.freq,
            fallback_model=self.fallback_model,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            test_size=test_size,
            input_size=input_size,
            level=level,
            refit=refit,
            fitted=fitted,
            prediction_intervals=prediction_intervals,
        )

    def _is_native(self, df) -> bool:
        engine = try_get_context_execution_engine()
        return engine is None and (df is None or isinstance(df, pd.DataFrame) or isinstance(df, pl.DataFrame))
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(StatsForecast, title_level=2, name='StatsForecast')
```

</details>
<details>
<summary>Code</summary>

``` python
# StatsForecast's class usage example

#from statsforecast.core import StatsForecast
from statsforecast.models import ( 
    ADIDA,
    AutoARIMA,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    HistoricAverage,
    IMAPA,
    Naive,
    RandomWalkWithDrift,
    SeasonalExponentialSmoothing,
    SeasonalNaive,
    SeasonalWindowAverage,
    SimpleExponentialSmoothing,
    TSB,
    WindowAverage,
    DynamicOptimizedTheta,
    AutoETS,
    AutoCES
)

# Generate synthetic panel DataFrame for example
panel_df = generate_series(n_series=9, equal_ends=False, engine='pandas')
panel_df.groupby('unique_id').tail(4)
```

</details>
<details>
<summary>Code</summary>

``` python
# Declare list of instantiated StatsForecast estimators to be fitted
# You can try other estimator's hyperparameters
# You can try other methods from the `models.StatsForecast` collection
# Check them here: https://nixtla.github.io/statsforecast/models.html
models=[AutoARIMA(), Naive(), 
        AutoETS(), AutoARIMA(allowmean=True, alias='MeanAutoARIMA')] 

# Instantiate StatsForecast class
fcst = StatsForecast(df=panel_df,
                     models=models,
                     freq='D', 
                     n_jobs=1, 
                     verbose=True)

# Efficiently predict
fcsts_df = fcst.forecast(h=4, fitted=True)
fcsts_df.groupby('unique_id').tail(4)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test custom names
test_eq(
    fcsts_df.columns[-1],
    'MeanAutoARIMA'
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test no duplicate names
test_fail(lambda: StatsForecast(models=[Naive(), Naive()], freq="D"))
StatsForecast(models=[Naive(), Naive(alias="Naive2")], freq="D")
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fig = StatsForecast.plot(panel_df, max_insample_length=10)
fig.show()
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_fail(
    StatsForecast.plot, 
    contains='Please use a list',
    kwargs={'df': panel_df, 'level': 90}
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df, fcsts_df, engine='matplotlib')
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test plot with ds as object
panel_df['ds'] = panel_df['ds'].astype(str)
fcst.plot(panel_df, fcsts_df)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcsts_df = fcst.forecast(h=4, fitted=True, level=[90, 80, 30])
fcsts_df.groupby('unique_id').tail(4)
fcst.plot(panel_df.groupby('unique_id').tail(28), fcsts_df, models=['AutoARIMA', 'AutoETS'], level=[90, 80])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(fcst.forecast_fitted_values(),
          forecasts_df=fcsts_df,
          models=['AutoARIMA', 'AutoETS'], level=[80], 
          max_insample_length=20,
          plot_anomalies=True)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df, fcsts_df, models=['AutoARIMA', 'Naive'])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df, fcsts_df, models=['AutoARIMA', 'Naive'], max_insample_length=28)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df.query('unique_id in [0, 1]'), fcsts_df, models=['AutoARIMA', 'Naive'], level=[90])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df, unique_ids=[0, 1], models=['AutoARIMA', 'Naive'], level=[90])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df.query('unique_id == 0'), fcsts_df, models=['AutoARIMA', 'Naive'], level=[90])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df, fcsts_df.query('unique_id == 0'), models=['AutoARIMA', 'Naive'], level=[90])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df, fcsts_df, unique_ids=[0], models=['AutoARIMA', 'Naive'], level=[90])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df.query('unique_id in [0, 1, 3]'), models=['AutoARIMA', 'Naive'], level=[90])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.plot(panel_df, fcsts_df, unique_ids=[0, 1, 2], level=[90])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fig = fcst.plot(panel_df, fcsts_df, unique_ids=[0, 1, 2, 3, 4], models=['AutoARIMA', 'Naive'], level=[90])
fig.show()
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fig = fcst.plot(
    panel_df, fcsts_df, unique_ids=[0, 1, 2, 3, 4], 
    models=['AutoARIMA', 'Naive'], 
    level=[90],
    engine='matplotlib'
)
fig
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test model prediction_interval overrides
models=[SimpleExponentialSmoothing(alpha=0.1, prediction_intervals=ConformalIntervals(h=24, n_windows=2))]
fcst = StatsForecast(df=panel_df,
                    models=models,
                    freq='D', 
                    n_jobs=1)
fcst._set_prediction_intervals(None)
assert models[0].prediction_intervals is not None
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst = StatsForecast(df=panel_df,
                     models=[AutoARIMA(season_length=7)],
                     freq='D', 
                     n_jobs=1, 
                     verbose=True)
fcsts_df = fcst.forecast(h=4, fitted=True, level=[90])
fcsts_df.groupby('unique_id').tail(4)
fitted_vals = fcst.forecast_fitted_values()
fcst.plot(panel_df, fitted_vals.drop(columns='y'), level=[90])
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.fit, 
         title_level=2,
         name='StatsForecast.fit')
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.predict, 
         title_level=2,
         name='SatstForecast.predict')
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.fit_predict, 
         title_level=2,
         name='StatsForecast.fit_predict')
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.forecast, title_level=2, name='StatsForecast.forecast')
```

</details>
<details>
<summary>Code</summary>

``` python
# StatsForecast.forecast method usage example

#from statsforecast.core import StatsForecast
from statsforecast.utils import AirPassengersDF as panel_df
from statsforecast.models import AutoARIMA, Naive

# Instantiate StatsForecast class
fcst = StatsForecast(df=panel_df,
                     models=[AutoARIMA(), Naive()],
                     freq='D', n_jobs=1)

# Efficiently predict without storing memory
fcsts_df = fcst.forecast(h=4, fitted=True)
fcsts_df.groupby('unique_id').tail(4)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
series = generate_series(100, n_static_features=2, equal_ends=False)

models = [
    ADIDA(), CrostonClassic(), CrostonOptimized(),
    CrostonSBA(), HistoricAverage(), 
    IMAPA(), Naive(), 
    RandomWalkWithDrift(), 
    SeasonalExponentialSmoothing(season_length=7, alpha=0.1),
    SeasonalNaive(season_length=7),
    SeasonalWindowAverage(season_length=7, window_size=4),
    SimpleExponentialSmoothing(alpha=0.1),
    TSB(alpha_d=0.1, alpha_p=0.3),
    WindowAverage(window_size=4)
]

fcst = StatsForecast(
    df=series,
    models=models,
    freq='D',
    n_jobs=1,
    verbose=True
)

res = fcst.forecast(h=14)

fcst_no_idx = StatsForecast(
    df=series.reset_index(),
    models=models,
    freq='D',
)
test_eq(
    fcst_no_idx.forecast(h=14),
    res
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test series without ds as datetime
series_wo_dt = series.copy()
series_wo_dt['ds'] = series_wo_dt['ds'].astype(str) 
fcst = StatsForecast(df=series_wo_dt,
                     models=models,
                     freq='D')
fcsts_wo_dt = fcst.forecast(h=14)
test_eq(res, fcsts_wo_dt)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_eq(res.index.unique(), fcst.uids)
last_dates = series.groupby('unique_id')['ds'].max()
test_eq(res.groupby('unique_id')['ds'].min().values, last_dates + pd.offsets.Day())
test_eq(res.groupby('unique_id')['ds'].max().values, last_dates + 14 * pd.offsets.Day())
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#tests for monthly data
monthly_series = generate_series(10_000, freq='M', min_length=10, max_length=20, equal_ends=True)
monthly_series

fcst = StatsForecast(
    models=[Naive()],
    freq='M'
)
monthly_res = fcst.forecast(df=monthly_series, h=4)
monthly_res

last_dates = monthly_series.groupby('unique_id')['ds'].max()
test_eq(monthly_res.groupby('unique_id')['ds'].min().values, pd.Series(fcst.last_dates) + pd.offsets.MonthEnd())
test_eq(monthly_res.groupby('unique_id')['ds'].max().values, pd.Series(fcst.last_dates) + 4 * pd.offsets.MonthEnd())
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.forecast_fitted_values, 
         title_level=2, 
         name='StatsForecast.forecast_fitted_values')
```

</details>
<details>
<summary>Code</summary>

``` python
# StatsForecast.forecast_fitted_values method usage example

#from statsforecast.core import StatsForecast
from statsforecast.utils import AirPassengersDF as panel_df
from statsforecast.models import Naive

# Instantiate StatsForecast class
fcst = StatsForecast(df=panel_df,
                     models=[AutoARIMA()],
                     freq='D', n_jobs=1)

# Access insample predictions
fcsts_df = fcst.forecast(h=12, fitted=True, level=(90, 10))
insample_fcsts_df = fcst.forecast_fitted_values()
insample_fcsts_df.tail(4)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#tests for fitted values
def test_fcst_fitted(series, n_jobs=1, str_ds=False):
    if str_ds:
        series = series.copy()
        series['ds'] = series['ds'].astype(str)
    fitted_fcst = StatsForecast(
        df=series,
        models=[Naive()],
        freq='D',
        n_jobs=n_jobs,
    )
    fitted_res = fitted_fcst.forecast(14, fitted=True)
    fitted = fitted_fcst.forecast_fitted_values()
    if str_ds:
        test_eq(pd.to_datetime(series['ds']), fitted['ds'])
    else:
        test_eq(series['ds'], fitted['ds'])
    test_eq(series['y'].astype(np.float32), fitted['y'])
test_fcst_fitted(series)
test_fcst_fitted(series, str_ds=True)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#tests for fallback model
def test_fcst_fallback_model(n_jobs=1):
    fitted_fcst = StatsForecast(
        df=series,
        models=[NullModel()],
        freq='D',
        n_jobs=n_jobs,
        fallback_model=Naive()
    )
    fitted_res = fitted_fcst.forecast(14, fitted=True)
    fitted = fitted_fcst.forecast_fitted_values()
    test_eq(series['ds'], fitted['ds'])
    test_eq(series['y'].astype(np.float32), fitted['y'])
    # test NullModel actualy fails
    fitted_fcst = StatsForecast(
        df=series,
        models=[NullModel()],
        freq='D',
        n_jobs=n_jobs,
    )
    test_fail(fitted_fcst.forecast, kwargs={'h': 14})
test_fcst_fallback_model()
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.cross_validation, 
         title_level=2, 
         name='StatsForecast.cross_validation')
```

</details>
<details>
<summary>Code</summary>

``` python
# StatsForecast.crossvalidation method usage example

#from statsforecast.core import StatsForecast
from statsforecast.utils import AirPassengersDF as panel_df
from statsforecast.models import Naive

# Instantiate StatsForecast class
fcst = StatsForecast(df=panel_df,
                     models=[Naive()],
                     freq='D', n_jobs=1, verbose=True)

# Access insample predictions
rolled_fcsts_df = fcst.cross_validation(14, n_windows=2)
rolled_fcsts_df.head(4)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#test for cross_validation
series_cv = pd.DataFrame({
    'ds': np.hstack([
        pd.date_range(end='2021-01-01', freq='D', periods=10),
        pd.date_range(end='2022-01-01', freq='D', periods=100),
        pd.date_range(end='2020-01-01', freq='D', periods=20)
    ]),
    'y': np.hstack([np.arange(10.), np.arange(100, 200), np.arange(20, 40)])
}, index=pd.Index(
    data=np.hstack([np.zeros(10), np.zeros(100) + 1, np.zeros(20) + 2]),
    name='unique_id'
))

fcst = StatsForecast(
    df=series_cv,
    models=[SumAhead(), Naive()],
    freq='D',
    verbose=True
)
res_cv = fcst.cross_validation(h=2, test_size=5, n_windows=None, level=(50, 60))
test_eq(0., np.mean(res_cv['y'] - res_cv['SumAhead']))

n_windows = fcst.cross_validation(h=2, n_windows=2).groupby('unique_id').size().unique()
test_eq(n_windows, 2 * 2)
test_eq(0., np.mean(res_cv['y'] - res_cv['SumAhead']))

n_windows = fcst.cross_validation(h=3, n_windows=3, step_size=3, fitted=True).groupby('unique_id').size().unique()
test_eq(n_windows, 3 * 3)
test_eq(0., np.mean(res_cv['y'] - res_cv['SumAhead']))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test cross validation refit=False
fcst = StatsForecast(
    df=series_cv,
    models=[SumAhead()],
    freq='D',
    verbose=True
)
res_cv_wo_refit = fcst.cross_validation(h=2, test_size=5, n_windows=None, level=(50, 60), refit=False)
test_fail(test_eq, args=(res_cv_wo_refit, res_cv))
cols_wo_refit = res_cv_wo_refit.columns
test_eq(res_cv_wo_refit.groupby('unique_id').head(1), 
        res_cv[cols_wo_refit].groupby('unique_id').head(1))

n_windows = fcst.cross_validation(h=2, n_windows=2, refit=False).groupby('unique_id').size().unique()
test_eq(n_windows, 2 * 2)

n_windows = fcst.cross_validation(h=3, n_windows=3, step_size=3, fitted=True, refit=False).groupby('unique_id').size().unique()
test_eq(n_windows, 3 * 3)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test cross validation refit=False
fcst = StatsForecast(
    df=series_cv,
    models=[DynamicOptimizedTheta(), AutoCES(), 
            DynamicOptimizedTheta(season_length=7, alias='test')],
    freq='D',
    verbose=True
)
res_cv_wo_refit = fcst.cross_validation(h=2, test_size=5, n_windows=None, level=(50, 60), refit=False)
res_cv_w_refit = fcst.cross_validation(h=2, test_size=5, n_windows=None, level=(50, 60), refit=True)
test_fail(test_eq, args=(res_cv_wo_refit, res_cv_w_refit))
test_eq(
    res_cv_wo_refit.groupby('unique_id').head(1), 
    res_cv_w_refit.groupby('unique_id').head(1)
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test series without ds as datetime
series_cv_wo_dt = series_cv.copy()
series_cv_wo_dt['ds'] = series_cv_wo_dt['ds'].astype(str) 
fcst = StatsForecast(
    df=series_cv_wo_dt,
    models=[SumAhead(), Naive()],
    freq='D',
    verbose=False
)
res_cv_wo_dt = fcst.cross_validation(h=2, test_size=5, n_windows=None, level=(50, 60))
test_eq(res_cv, res_cv_wo_dt)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#test for equal ends cross_validation
series_cv = pd.DataFrame({
    'ds': np.hstack([
        pd.date_range(end='2022-01-01', freq='D', periods=10),
        pd.date_range(end='2022-01-01', freq='D', periods=100),
        pd.date_range(end='2022-01-01', freq='D', periods=20)
    ]),
    'y': np.hstack([np.arange(10), np.arange(100, 200), np.arange(20, 40)])
}, index=pd.Index(
    data=np.hstack([np.zeros(10), np.zeros(100) + 1, np.zeros(20) + 2]),
    name='unique_id'
))
fcst = StatsForecast(
    models=[SumAhead()],
    freq='D',
)
res_cv = fcst.cross_validation(df=series_cv, h=2, test_size=5, n_windows=None, level=(50,60), fitted=True)
test_eq(0., np.mean(res_cv['y'] - res_cv['SumAhead']))

n_windows = fcst.cross_validation(h=2, n_windows=2).groupby('unique_id').size().unique()
test_eq(n_windows, 2 * 2)
test_eq(0., np.mean(res_cv['y'] - res_cv['SumAhead']))

n_windows = fcst.cross_validation(h=3, n_windows=3, step_size=3).groupby('unique_id').size().unique()
test_eq(n_windows, 3 * 3)
test_eq(0., np.mean(res_cv['y'] - res_cv['SumAhead']))
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.cross_validation_fitted_values, 
         title_level=2, 
         name='StatsForecast.cross_validation_fitted_values')
```

</details>
<details>
<summary>Code</summary>

``` python
# StatsForecast.cross_validation_fitted_values method usage example

#from statsforecast.core import StatsForecast
from statsforecast.utils import AirPassengersDF as panel_df
from statsforecast.models import Naive

# Instantiate StatsForecast class
fcst = StatsForecast(df=panel_df,
                     models=[Naive()],
                     freq='D', n_jobs=1)

# Access insample predictions
rolled_fcsts_df = fcst.cross_validation(h=12, n_windows=2, fitted=True)
insample_rolled_fcsts_df = fcst.cross_validation_fitted_values()
insample_rolled_fcsts_df.tail(4)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#tests for fitted values cross_validation
def test_cv_fitted(series_cv, n_jobs=1, str_ds=False):
    if str_ds:
        series_cv = series_cv.copy()
        series_cv['ds'] = series_cv['ds'].astype(str)
    resids_fcst = StatsForecast(
        df=series_cv,
        models=[SumAhead(), Naive()],
        freq='D',
        n_jobs=n_jobs
    )
    resids_res_cv = resids_fcst.cross_validation(h=2, n_windows=4, fitted=True)
    resids_cv = resids_fcst.cross_validation_fitted_values()
    np.testing.assert_array_equal(
        resids_cv['cutoff'].unique(),
        resids_res_cv['cutoff'].unique()
    )
    if str_ds:
        series_cv['ds'] = pd.to_datetime(series_cv['ds'])
    for uid in resids_cv.index.unique():
        for cutoff in resids_cv.loc[uid]['cutoff'].unique():
            pd.testing.assert_frame_equal(
                resids_cv.loc[uid].query('cutoff == @cutoff')[['ds', 'y']], 
                series_cv.query('ds <= @cutoff & unique_id == @uid')[['ds', 'y']],
                check_dtype=False
            )
test_cv_fitted(series_cv)
test_cv_fitted(series_cv, str_ds=True)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
#tests for fallback model
def test_cv_fallback_model(n_jobs=1):
    fitted_fcst = StatsForecast(
        df=series,
        models=[NullModel()],
        freq='D',
        n_jobs=n_jobs,
        fallback_model=Naive()
    )
    fitted_res = fitted_fcst.cross_validation(h=2, n_windows=4, fitted=True)
    fitted = fitted_fcst.cross_validation_fitted_values()
    # test NullModel actualy fails
    fitted_fcst = StatsForecast(
        df=series,
        models=[NullModel()],
        freq='D',
        n_jobs=n_jobs,
    )
    test_fail(fitted_fcst.cross_validation, 
              kwargs={'h': 2, 'n_windows': 4}, 
              contains='got an unexpected keyword argument')
test_cv_fallback_model()
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(_StatsForecast.plot, 
         title_level=2, 
         name='StatsForecast.plot')
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst = fcst.fit(df=series)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_eq(
    fcst.predict(h=12),
    fcst.forecast(h=12)
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_eq(
    fcst.fit_predict(h=12),
    fcst.forecast(h=12)
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test for conformal prediction
uids = series.index.unique()[:10]
series_subset = series.query('unique_id in @uids')[['ds', 'y']]
sf = StatsForecast(
    models=[AutoARIMA()],
    freq='D', 
    n_jobs=1,
)
sf = sf.fit(df=series_subset, prediction_intervals=ConformalIntervals(h=12))
test_eq(
    sf.predict(h=12, level=[80, 90]),
    sf.fit_predict(df=series_subset, h=12, level=[80, 90], prediction_intervals=ConformalIntervals(h=12)),
)
test_eq(
    sf.predict(h=12, level=[80, 90]),
    sf.forecast(df=series_subset, h=12, level=[80, 90], prediction_intervals=ConformalIntervals(h=12)),
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test conformal cross validation
cv_conformal = sf.cross_validation(
    df=series_subset, 
    h=12, 
    n_windows=2,
    level=[80, 90], 
    prediction_intervals=ConformalIntervals(h=12),
)
cv_no_conformal = sf.cross_validation(
    df=series_subset, 
    h=12, 
    n_windows=2,
    level=[80, 90], 
)
test_eq(
    cv_conformal.columns,
    cv_no_conformal.columns,
)
test_eq(
    cv_conformal.filter(regex='ds|cutoff|y|AutoARIMA$'),
    cv_no_conformal.filter(regex='ds|cutoff|y|AutoARIMA$')
)
```

</details>

:::

# Misc {#misc}

## Integer datestamp {#integer-datestamp}

The `StatsForecast` class can also receive integers as datestamp, the
following example shows how to do it.

<details>
<summary>Code</summary>

``` python
# from statsforecast.core import StatsForecast
from statsforecast.utils import AirPassengers as ap
from statsforecast.models import HistoricAverage
```

</details>
<details>
<summary>Code</summary>

``` python
int_ds_df = pd.DataFrame({'ds': np.arange(1, len(ap) + 1), 'y': ap})
int_ds_df.insert(0, 'unique_id', 'AirPassengers')
int_ds_df.set_index('unique_id', inplace=True)
int_ds_df.head()
```

</details>
<details>
<summary>Code</summary>

``` python
int_ds_df.tail()
```

</details>
<details>
<summary>Code</summary>

``` python
int_ds_df
```

</details>
<details>
<summary>Code</summary>

``` python
fcst = StatsForecast(df=int_ds_df, models=[HistoricAverage()], freq='D')
horizon = 7
forecast = fcst.forecast(horizon)
forecast.head()
```

</details>
<details>
<summary>Code</summary>

``` python
last_date = int_ds_df['ds'].max()
test_eq(forecast['ds'].values, np.arange(last_date + 1, last_date + 1 + horizon))
```

</details>
<details>
<summary>Code</summary>

``` python
int_ds_cv = fcst.cross_validation(h=7, test_size=8, n_windows=None)
int_ds_cv
```

</details>

## External regressors {#external-regressors}

Every column after **y** is considered an external regressor and will be
passed to the models that allow them. If you use them you must supply
the future values to the `StatsForecast.forecast` method.

<details>
<summary>Code</summary>

``` python
class LinearRegression:
    
    def __init__(self):
        pass
    
    def fit(self, y, X):
        self.coefs_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self
    
    def predict(self, h, X):
        mean = X @ coefs
        return mean
    
    def __repr__(self):
        return 'LinearRegression()'
    
    def forecast(self, y, h, X=None, X_future=None, fitted=False):
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        return {'mean': X_future @ coefs}
    
    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b
```

</details>
<details>
<summary>Code</summary>

``` python
series_xreg = series = generate_series(10_000, equal_ends=True)
series_xreg['intercept'] = 1
series_xreg['dayofweek'] = series_xreg['ds'].dt.dayofweek
series_xreg = pd.get_dummies(series_xreg, columns=['dayofweek'], drop_first=True)
series_xreg
```

</details>
<details>
<summary>Code</summary>

``` python
dates = sorted(series_xreg['ds'].unique())
valid_start = dates[-14]
train_mask = series_xreg['ds'] < valid_start
series_train = series_xreg[train_mask]
series_valid = series_xreg[~train_mask]
X_valid = series_valid.drop(columns=['y'])
fcst = StatsForecast(
    df=series_train,
    models=[LinearRegression()],
    freq='D',
)
xreg_res = fcst.forecast(14, X_df=X_valid)
xreg_res['y'] = series_valid['y'].values
```

</details>
<details>
<summary>Code</summary>

``` python
xreg_res.groupby('ds').mean().plot()
```

</details>
<details>
<summary>Code</summary>

``` python
xreg_res_cv = fcst.cross_validation(h=3, test_size=5, n_windows=None)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# the following cells contain tests for external regressors
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
class ReturnX:
    
    def __init__(self):
        pass
    
    def fit(self, y, X):
        return self
    
    def predict(self, h, X):
        mean = X
        return X
    
    def __repr__(self):
        return 'ReturnX'
    
    def forecast(self, y, h, X=None, X_future=None, fitted=False):
        return {'mean': X_future.flatten()}
    
    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
df = pd.DataFrame(
    {
        'ds': np.hstack([np.arange(10), np.arange(10)]),
        'y': np.random.rand(20),
        'x': np.arange(20, dtype=np.float32),
    },
    index=pd.Index([0] * 10 + [1] * 10, name='unique_id'),
)
train_mask = df['ds'] < 6
train_df = df[train_mask]
test_df = df[~train_mask]
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
def test_x_vars(n_jobs=1):
    fcst = StatsForecast(
        df=train_df,
        models=[ReturnX()],
        freq='M',
        n_jobs=n_jobs,
    )
    xreg = test_df.drop(columns='y')
    res = fcst.forecast(4, X_df=xreg)
    expected_res = xreg.rename(columns={'x': 'ReturnX'})
    pd.testing.assert_frame_equal(res, expected_res, check_dtype=False)
test_x_vars(n_jobs=1)
```

</details>

:::

## Prediction intervals {#prediction-intervals}

You can pass the argument `level` to the `StatsForecast.forecast` method
to calculate prediction intervals. Not all models can calculate them at
the moment, so we will only obtain the intervals of those models that
have it implemented.

<details>
<summary>Code</summary>

``` python
ap_df = pd.DataFrame({'ds': np.arange(ap.size), 'y': ap}, index=pd.Index([0] * ap.size, name='unique_id'))
sf = StatsForecast(
    models=[
        SeasonalNaive(season_length=12), 
        AutoARIMA(season_length=12)
    ],
    freq='M',
    n_jobs=1
)
ap_ci = sf.forecast(df=ap_df, h=12, level=(80, 95))
fcst.plot(ap_ci, level=[80])
```

</details>

## Conformal Prediction intervals {#conformal-prediction-intervals}

You can also add conformal intervals using the following code.

<details>
<summary>Code</summary>

``` python
from statsforecast.utils import ConformalIntervals
```

</details>
<details>
<summary>Code</summary>

``` python
sf = StatsForecast(
    models=[
        AutoARIMA(season_length=12),
        AutoARIMA(
            season_length=12, 
            prediction_intervals=ConformalIntervals(n_windows=2, h=12),
            alias='ConformalAutoARIMA'
        ),
    ],
    freq='M',
    n_jobs=1
)
ap_ci = sf.forecast(df=ap_df, h=12, level=(80, 95))
fcst.plot(ap_ci, level=[80])
```

</details>

You can also compute conformal intervals for all the models that support
them, using the following,

<details>
<summary>Code</summary>

``` python
sf = StatsForecast(
    models=[
        AutoARIMA(season_length=12),
    ],
    freq='M',
    n_jobs=1
)
ap_ci = sf.forecast(
    df=ap_df, 
    h=12, 
    level=(50, 80, 95), 
    prediction_intervals=ConformalIntervals(h=12),
)
fcst.plot(ap_ci, level=[50, 80, 95])
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
def test_conf_intervals(n_jobs=1):
    ap_df = pd.DataFrame({'ds': np.arange(ap.size), 'y': ap}, index=pd.Index([0] * ap.size, name='unique_id'))
    fcst = StatsForecast(
        df=ap_df,
        models=[
            SeasonalNaive(season_length=12), 
            AutoARIMA(season_length=12)
        ],
        freq='M',
        n_jobs=n_jobs
    )
    ap_ci = fcst.forecast(12, level=(80, 95))
    ap_ci.set_index('ds').plot(marker='.', figsize=(10, 6))
test_conf_intervals(n_jobs=1)
```

</details>

:::

