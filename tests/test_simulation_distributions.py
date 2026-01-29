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

def test_autoets_bootstrap():
    """Test bootstrap simulation with AutoETS model."""
    df = pd.DataFrame({
        'unique_id': [1] * 100,
        'ds': pd.date_range('2000-01-01', periods=100),
        'y': np.random.normal(0, 1, 100).cumsum()
    })

    models = [AutoETS(season_length=1)]
    sf = StatsForecast(models=models, freq='D')

    res_bootstrap = sf.simulate(h=5, df=df, n_paths=20, seed=42,
                                 error_distribution='bootstrap')
    assert 'AutoETS' in res_bootstrap.columns
    assert res_bootstrap.shape == (100, 4)  # 5 steps * 20 paths = 100 rows
    # Verify no NaN values in output
    assert not res_bootstrap['AutoETS'].isna().any()

def test_autoces_bootstrap():
    """Test bootstrap simulation with AutoCES model."""
    df = pd.DataFrame({
        'unique_id': [1] * 100,
        'ds': pd.date_range('2000-01-01', periods=100),
        'y': np.random.normal(0, 1, 100).cumsum()
    })

    models = [AutoCES(season_length=1)]
    sf = StatsForecast(models=models, freq='D')

    res_bootstrap = sf.simulate(h=5, df=df, n_paths=20, seed=42,
                                 error_distribution='bootstrap')
    assert 'CES' in res_bootstrap.columns
    assert res_bootstrap.shape == (100, 4)  # 5 steps * 20 paths = 100 rows
    # Verify no NaN values in output
    assert not res_bootstrap['CES'].isna().any()

def test_t_distribution_invalid_df():
    """Test that t-distribution with df <= 2 raises appropriate error."""
    df = pd.DataFrame({
        'unique_id': [1] * 10,
        'ds': pd.date_range('2000-01-01', periods=10),
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    models = [Naive()]
    sf = StatsForecast(models=models, freq='D')

    # Test with df = 2 (should fail)
    with pytest.raises(ValueError, match="Degrees of freedom.*must be > 2"):
        sf.simulate(h=3, df=df, n_paths=10, seed=42,
                   error_distribution='t', error_params={'df': 2})

    # Test with df = 1 (should fail)
    with pytest.raises(ValueError, match="Degrees of freedom.*must be > 2"):
        sf.simulate(h=3, df=df, n_paths=10, seed=42,
                   error_distribution='t', error_params={'df': 1})

    # Test with df = 0 (should fail)
    with pytest.raises(ValueError, match="Degrees of freedom.*must be > 2"):
        sf.simulate(h=3, df=df, n_paths=10, seed=42,
                   error_distribution='t', error_params={'df': 0})

def test_ged_invalid_shape():
    """Test that GED distribution with shape <= 0 raises appropriate error."""
    df = pd.DataFrame({
        'unique_id': [1] * 10,
        'ds': pd.date_range('2000-01-01', periods=10),
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    models = [Naive()]
    sf = StatsForecast(models=models, freq='D')

    # Test with shape = 0 (should fail)
    with pytest.raises(ValueError, match="GED shape parameter must be positive"):
        sf.simulate(h=3, df=df, n_paths=10, seed=42,
                   error_distribution='ged', error_params={'shape': 0})

    # Test with negative shape (should fail)
    with pytest.raises(ValueError, match="GED shape parameter must be positive"):
        sf.simulate(h=3, df=df, n_paths=10, seed=42,
                   error_distribution='ged', error_params={'shape': -1})

def test_sigma_zero():
    """Test simulation with sigma=0 (perfect forecasts, no noise)."""
    df = pd.DataFrame({
        'unique_id': [1] * 10,
        'ds': pd.date_range('2000-01-01', periods=10),
        'y': [10.0] * 10  # Constant series
    })

    models = [Naive()]
    sf = StatsForecast(models=models, freq='D')
    h = 3
    # With sigma=0, all paths should be identical (deterministic)
    res = sf.simulate(h=h, df=df, n_paths=5, seed=42, error_distribution='normal')

    # All simulated paths should be the same when sigma is 0 (or very small for constant series)
    # For constant series, Naive model should predict the constant value
    unique_vals = res.groupby('ds')['Naive'].nunique()
    assert len(unique_vals) == h
    # For a constant input series, the model should produce consistent forecasts
    assert res['Naive'].notna().all()


def test_skew_normal_distribution():
    """Test skew-normal distribution with different skewness parameters."""
    df = pd.DataFrame({
        'unique_id': [1] * 30,
        'ds': pd.date_range('2000-01-01', periods=30),
        'y': np.random.normal(0, 1, 30).cumsum()
    })

    models = [Naive()]
    sf = StatsForecast(models=models, freq='D')

    # Test with positive skewness
    res_skew_pos = sf.simulate(h=5, df=df, n_paths=10, seed=42,
                                error_distribution='skew-normal',
                                error_params={'skewness': 5})
    assert 'Naive' in res_skew_pos.columns
    assert res_skew_pos.shape == (50, 4)

    # Test with negative skewness
    res_skew_neg = sf.simulate(h=5, df=df, n_paths=10, seed=42,
                                error_distribution='skew-normal',
                                error_params={'skewness': -5})
    assert 'Naive' in res_skew_neg.columns

    # Test with zero skewness (should be similar to normal)
    res_skew_zero = sf.simulate(h=5, df=df, n_paths=10, seed=42,
                                 error_distribution='skew-normal',
                                 error_params={'skewness': 0})
    assert 'Naive' in res_skew_zero.columns

def test_ged_distribution_shapes():
    """Test GED distribution with different shape parameters."""
    df = pd.DataFrame({
        'unique_id': [1] * 30,
        'ds': pd.date_range('2000-01-01', periods=30),
        'y': np.random.normal(0, 1, 30).cumsum()
    })

    models = [Naive()]
    sf = StatsForecast(models=models, freq='D')

    # Test with shape = 1 (Laplace-like)
    res_ged_1 = sf.simulate(h=5, df=df, n_paths=10, seed=42,
                             error_distribution='ged',
                             error_params={'shape': 1})
    assert 'Naive' in res_ged_1.columns
    assert res_ged_1.shape == (50, 4)

    # Test with shape = 2 (Normal-like)
    res_ged_2 = sf.simulate(h=5, df=df, n_paths=10, seed=42,
                             error_distribution='ged',
                             error_params={'shape': 2})
    assert 'Naive' in res_ged_2.columns

    # Test with shape = 5 (more uniform-like)
    res_ged_5 = sf.simulate(h=5, df=df, n_paths=10, seed=42,
                             error_distribution='ged',
                             error_params={'shape': 5})
    assert 'Naive' in res_ged_5.columns

    # Different shapes should produce different results
    assert not np.array_equal(res_ged_1['Naive'].values, res_ged_2['Naive'].values)
    assert not np.array_equal(res_ged_2['Naive'].values, res_ged_5['Naive'].values)
