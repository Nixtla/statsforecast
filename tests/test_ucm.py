"""
Tests for UCM (Unobserved Components Model) wrapper.

Run with: pytest test_ucm.py -v
"""

import numpy as np
import pytest


def test_ucm_import():
    """Test that UCM classes can be imported."""
    from ucm import UCM, LocalLevel, LocalLinearTrend, SmoothTrend, RandomWalkWithDrift
    assert UCM is not None
    assert LocalLevel is not None
    assert LocalLinearTrend is not None
    assert SmoothTrend is not None
    assert RandomWalkWithDrift is not None


def test_ucm_basic_fit_predict():
    """Test basic fit and predict workflow."""
    from ucm import UCM
    
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 50
    
    model = UCM(level='local level')
    model.fit(y)
    
    forecast = model.predict(h=10)
    
    assert 'mean' in forecast
    assert len(forecast['mean']) == 10
    assert not np.any(np.isnan(forecast['mean']))


def test_ucm_with_seasonal():
    """Test UCM with seasonal component."""
    from ucm import UCM
    
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    y = 50 + 0.3 * t + 8 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2
    
    model = UCM(level='local level', seasonal=12)
    model.fit(y)
    
    forecast = model.predict(h=12)
    
    assert len(forecast['mean']) == 12


def test_ucm_prediction_intervals():
    """Test prediction intervals."""
    from ucm import UCM
    
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 50
    
    model = UCM(level='local level')
    model.fit(y)
    
    forecast = model.predict(h=10, level=[80, 95])
    
    assert 'mean' in forecast
    assert 'lo-80' in forecast
    assert 'hi-80' in forecast
    assert 'lo-95' in forecast
    assert 'hi-95' in forecast
    
    # Check that intervals make sense
    assert np.all(forecast['lo-95'] <= forecast['lo-80'])
    assert np.all(forecast['lo-80'] <= forecast['mean'])
    assert np.all(forecast['mean'] <= forecast['hi-80'])
    assert np.all(forecast['hi-80'] <= forecast['hi-95'])


def test_ucm_forecast_method():
    """Test the forecast() method (fit + predict in one call)."""
    from ucm import UCM
    
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 50
    
    model = UCM(level='local linear trend')
    result = model.forecast(y=y, h=10, level=[90], fitted=True)
    
    assert 'mean' in result
    assert 'fitted' in result
    assert 'lo-90' in result
    assert 'hi-90' in result
    assert len(result['mean']) == 10
    assert len(result['fitted']) == 100


def test_ucm_predict_in_sample():
    """Test in-sample predictions."""
    from ucm import UCM
    
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 50
    
    model = UCM(level='local level')
    model.fit(y)
    
    insample = model.predict_in_sample(level=[95])
    
    assert 'fitted' in insample
    assert len(insample['fitted']) == 100
    assert 'lo-95' in insample
    assert 'hi-95' in insample


def test_ucm_get_components():
    """Test component extraction."""
    from ucm import UCM
    
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    y = 50 + 0.3 * t + 8 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2
    
    model = UCM(level='local level', seasonal=12)
    model.fit(y)
    
    components = model.get_components()
    
    assert 'level' in components
    assert 'seasonal' in components
    assert len(components['level']) == n
    assert len(components['seasonal']) == n


def test_ucm_with_exogenous():
    """Test UCM with exogenous variables."""
    from ucm import UCM
    
    np.random.seed(42)
    n = 100
    y = np.cumsum(np.random.randn(n)) + 50
    X = np.column_stack([np.ones(n), np.arange(n)])
    X_future = np.column_stack([np.ones(10), np.arange(n, n+10)])
    
    model = UCM(level='local level')
    model.fit(y, X=X)
    
    forecast = model.predict(h=10, X=X_future)
    
    assert len(forecast['mean']) == 10


def test_local_level():
    """Test LocalLevel convenience class."""
    from ucm import LocalLevel
    
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 50
    
    model = LocalLevel()
    model.fit(y)
    
    assert 'LocalLevel' in model.alias
    
    forecast = model.predict(h=5)
    assert len(forecast['mean']) == 5


def test_local_linear_trend():
    """Test LocalLinearTrend convenience class."""
    from ucm import LocalLinearTrend
    
    np.random.seed(42)
    n = 100
    y = 50 + 0.5 * np.arange(n) + np.random.randn(n) * 2
    
    model = LocalLinearTrend()
    model.fit(y)
    
    assert 'LocalLinearTrend' in model.alias
    
    forecast = model.predict(h=5)
    assert len(forecast['mean']) == 5


def test_smooth_trend():
    """Test SmoothTrend convenience class."""
    from ucm import SmoothTrend
    
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 50
    
    model = SmoothTrend()
    model.fit(y)
    
    assert 'SmoothTrend' in model.alias


def test_ucm_alias():
    """Test custom alias."""
    from ucm import UCM
    
    model = UCM(level='local level', alias='MyCustomModel')
    assert str(model) == 'MyCustomModel'
    assert model.alias == 'MyCustomModel'


def test_ucm_new():
    """Test the new() method for copying."""
    from ucm import UCM
    
    model1 = UCM(level='local level', seasonal=12)
    model2 = model1.new()
    
    assert model2.level == model1.level
    assert model2.seasonal == model1.seasonal
    assert model2 is not model1


def test_statsforecast_integration():
    """Test integration with StatsForecast."""
    pytest.importorskip("statsforecast")
    
    import pandas as pd
    from statsforecast import StatsForecast
    from ucm import LocalLevel, LocalLinearTrend
    
    np.random.seed(42)
    n = 60
    dates = pd.date_range('2020-01-01', periods=n, freq='MS')
    y = 50 + 0.3 * np.arange(n) + np.random.randn(n) * 2
    
    df = pd.DataFrame({
        'unique_id': ['series1'] * n,
        'ds': dates,
        'y': y
    })
    
    sf = StatsForecast(
        models=[LocalLevel(), LocalLinearTrend()],
        freq='MS',
        n_jobs=1,
    )
    
    sf.fit(df)
    forecast = sf.predict(h=6)
    
    assert len(forecast) == 6
    assert 'LocalLevel' in forecast.columns
    assert 'LocalLinearTrend' in forecast.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
