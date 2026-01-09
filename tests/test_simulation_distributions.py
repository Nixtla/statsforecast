import numpy as np
import pandas as pd
import pytest
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive, 
    HistoricAverage, RandomWalkWithDrift, SimpleExponentialSmoothing
)

def test_simulation_distributions_basic():
    df = pd.DataFrame({
        'unique_id': [1, 1, 1, 1, 1],
        'ds': pd.date_range('2000-01-01', periods=5),
        'y': [1, 2, 3, 4, 10]
    })
    
    models = [Naive()]
    sf = StatsForecast(models=models, freq='D')
    
    # Test normal (default)
    res_normal = sf.simulate(h=3, df=df, n_paths=10, seed=42)
    assert 'Naive' in res_normal.columns
    assert res_normal.shape == (30, 4) # 3 steps * 10 paths = 30 rows
    
    # Test Student's t
    res_t = sf.simulate(h=3, df=df, n_paths=10, seed=42, 
                        error_distribution='t', error_params={'df': 3})
    assert 'Naive' in res_t.columns
    
    # Test Laplace
    res_laplace = sf.simulate(h=3, df=df, n_paths=10, seed=42, 
                              error_distribution='laplace')
    assert 'Naive' in res_laplace.columns
    
    # Ensure different distributions produce different results with same seed (if sigma is same)
    # Actually, they might not be wildly different for h=1, but let's check values
    assert not np.array_equal(res_normal['Naive'].values, res_t['Naive'].values)
    assert not np.array_equal(res_normal['Naive'].values, res_laplace['Naive'].values)

def test_simulation_bootstrap():
    df = pd.DataFrame({
        'unique_id': [1] * 100,
        'ds': pd.date_range('2000-01-01', periods=100),
        'y': np.random.normal(0, 1, 100).cumsum()
    })
    
    # Use ARIMA for bootstrap as it stores residuals
    models = [AutoARIMA()]
    sf = StatsForecast(models=models, freq='D')
    
    res_bootstrap = sf.simulate(h=5, df=df, n_paths=20, seed=42, 
                                 error_distribution='bootstrap')
    assert 'AutoARIMA' in res_bootstrap.columns
    assert res_bootstrap.shape == (100, 4) # 5 steps * 20 paths = 100 rows

def test_simulation_all_models_distributions():
    df = pd.DataFrame({
        'unique_id': [1] * 20,
        'ds': pd.date_range('2000-01-01', periods=20),
        'y': np.random.normal(0, 1, 20).cumsum()
    })
    
    models = [
        AutoARIMA(),
        AutoETS(season_length=4),
        AutoCES(season_length=4),
        AutoTheta(season_length=4),
        Naive(),
        HistoricAverage(),
        RandomWalkWithDrift(),
        SimpleExponentialSmoothing(alpha=0.5)
    ]
    sf = StatsForecast(models=models, freq='D')
    
    # Test common distribution for all
    res = sf.simulate(h=2, df=df, n_paths=5, seed=42, 
                      error_distribution='t', error_params={'df': 5})
    
    for model in models:
        assert repr(model) in res.columns

def test_invalid_distribution():
    df = pd.DataFrame({
        'unique_id': [1, 2],
        'ds': [1, 1],
        'y': [1, 2]
    })
    sf = StatsForecast(models=[Naive()], freq=1)
    with pytest.raises(ValueError, match="Unsupported error distribution"):
        sf.simulate(h=1, df=df, error_distribution='invalid_dist')

def test_reproducibility_with_distributions():
    df = pd.DataFrame({
        'unique_id': [1] * 10,
        'ds': range(10),
        'y': np.random.randn(10)
    })
    sf = StatsForecast(models=[Naive()], freq=1)
    
    res1 = sf.simulate(h=5, df=df, n_paths=10, seed=42, error_distribution='t', error_params={'df': 3})
    res2 = sf.simulate(h=5, df=df, n_paths=10, seed=42, error_distribution='t', error_params={'df': 3})
    
    np.testing.assert_array_almost_equal(res1['Naive'].values, res2['Naive'].values)
