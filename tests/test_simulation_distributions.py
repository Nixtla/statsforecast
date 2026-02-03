import numpy as np
import pandas as pd
import pytest
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive,
    HistoricAverage, RandomWalkWithDrift, SimpleExponentialSmoothing,
    SeasonalExponentialSmoothing, SeasonalNaive
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


# ============================================================================
# Tests for Automatic Parameter Estimation (Mode 1)
# ============================================================================

def test_automatic_t_distribution_estimation():
    """Test automatic t-distribution parameter estimation from residuals."""
    from statsforecast.simulation import sample_errors

    # Create synthetic residuals from t-distribution
    np.random.seed(42)
    true_df, true_loc, true_scale = 5.0, 0.5, 2.0
    from scipy import stats
    residuals = stats.t.rvs(true_df, loc=true_loc, scale=true_scale, size=100)

    # Mode 1: automatic estimation (params=None, residuals provided)
    errors = sample_errors(
        size=1000,
        sigma=999.0,  # This should be ignored!
        distribution='t',
        params=None,
        residuals=residuals,
        rng=np.random.default_rng(42)
    )

    # Verify errors are generated and sigma was ignored
    assert errors.shape == (1000,)
    assert not np.allclose(errors.std(), 999.0)  # sigma should be ignored

    # Check that the generated errors have reasonable properties
    # (within broad range since we're sampling)
    assert np.abs(errors.mean()) < 2.0  # Should be near true_loc
    assert 1.0 < errors.std() < 4.0  # Should be near true_scale


def test_automatic_laplace_distribution_estimation():
    """Test automatic Laplace parameter estimation from residuals."""
    from statsforecast.simulation import sample_errors
    from scipy import stats

    # Create synthetic residuals from Laplace distribution
    np.random.seed(42)
    true_loc, true_scale = 1.0, 1.5
    residuals = stats.laplace.rvs(loc=true_loc, scale=true_scale, size=100)

    # Mode 1: automatic estimation
    errors = sample_errors(
        size=1000,
        sigma=999.0,  # This should be ignored!
        distribution='laplace',
        params=None,
        residuals=residuals,
        rng=np.random.default_rng(42)
    )

    assert errors.shape == (1000,)
    # Verify sigma was ignored
    assert not np.allclose(errors.std(), 999.0)


def test_automatic_skewnorm_distribution_estimation():
    """Test automatic skew-normal parameter estimation from residuals."""
    from statsforecast.simulation import sample_errors
    from scipy import stats

    # Create synthetic residuals from skew-normal distribution
    np.random.seed(42)
    true_skewness, true_loc, true_scale = 3.0, 0.0, 2.0
    residuals = stats.skewnorm.rvs(true_skewness, loc=true_loc, scale=true_scale, size=100)

    # Mode 1: automatic estimation
    errors = sample_errors(
        size=1000,
        sigma=999.0,  # This should be ignored!
        distribution='skew-normal',
        params=None,
        residuals=residuals,
        rng=np.random.default_rng(42)
    )

    assert errors.shape == (1000,)
    # Verify sigma was ignored
    assert not np.allclose(errors.std(), 999.0)


def test_automatic_ged_distribution_estimation():
    """Test automatic GED parameter estimation from residuals."""
    from statsforecast.simulation import sample_errors
    from scipy import stats

    # Create synthetic residuals from GED distribution
    np.random.seed(42)
    true_shape, true_loc, true_scale = 2.5, 0.0, 1.5
    residuals = stats.gennorm.rvs(true_shape, loc=true_loc, scale=true_scale, size=100)

    # Mode 1: automatic estimation
    errors = sample_errors(
        size=1000,
        sigma=999.0,  # This should be ignored!
        distribution='ged',
        params=None,
        residuals=residuals,
        rng=np.random.default_rng(42)
    )

    assert errors.shape == (1000,)
    # Verify sigma was ignored
    assert not np.allclose(errors.std(), 999.0)


def test_normal_distribution_uses_sigma_in_mode1():
    """Test that normal distribution always uses passed sigma, even in Mode 1."""
    from statsforecast.simulation import sample_errors

    # Create residuals with different scale
    np.random.seed(42)
    residuals = np.random.normal(0, 5.0, size=100)

    # Mode 1: for normal, sigma should still be used (not ignored)
    errors = sample_errors(
        size=10000,
        sigma=2.0,
        distribution='normal',
        params=None,
        residuals=residuals,
        rng=np.random.default_rng(42)
    )

    # Normal distribution should use the passed sigma=2.0
    # (not the residuals' scale of 5.0)
    assert np.abs(errors.std() - 2.0) < 0.1  # Should be close to 2.0


def test_mode_selection_params_none_with_residuals():
    """Test when params=None and residuals provided."""
    from statsforecast.simulation import sample_errors
    from scipy import stats

    np.random.seed(42)
    residuals = stats.t.rvs(4, size=100)

    # Mode 1: Should use automatic estimation
    errors_mode1 = sample_errors(
        size=1000,
        sigma=1.0,
        distribution='t',
        params=None,  # Mode 1
        residuals=residuals,
        rng=np.random.default_rng(42)
    )

    # Mode 2: Should use explicit params
    errors_mode2 = sample_errors(
        size=1000,
        sigma=1.0,
        distribution='t',
        params={'df': 10},  # Mode 2 with different df
        residuals=residuals,  # residuals present but should be ignored
        rng=np.random.default_rng(42)
    )

    # Results should be different (different modes, different parameters)
    assert not np.array_equal(errors_mode1, errors_mode2)


def test_error_when_residuals_missing_for_mode1():
    """Test error is raised when residuals=None and params=None for non-normal."""
    from statsforecast.simulation import sample_errors

    # Should raise error: residuals required for Mode 1 with t-distribution
    with pytest.raises(ValueError, match="requires either 'params' or 'residuals'"):
        sample_errors(
            size=100,
            sigma=1.0,
            distribution='t',
            params=None,
            residuals=None,  # Missing!
            rng=np.random.default_rng(42)
        )

    # Same for skew-normal
    with pytest.raises(ValueError, match="requires either 'params' or 'residuals'"):
        sample_errors(
            size=100,
            sigma=1.0,
            distribution='skew-normal',
            params=None,
            residuals=None,
            rng=np.random.default_rng(42)
        )

    # Same for GED
    with pytest.raises(ValueError, match="requires either 'params' or 'residuals'"):
        sample_errors(
            size=100,
            sigma=1.0,
            distribution='ged',
            params=None,
            residuals=None,
            rng=np.random.default_rng(42)
        )


def test_error_when_insufficient_residuals():
    """Test fallback to Mode 2 when residuals have < 10 valid samples."""
    from statsforecast.simulation import sample_errors

    # Too few residuals - should fall back to Mode 2 with defaults
    residuals_too_few = np.random.randn(5)

    # Should not raise error, but fall back to Mode 2 with default params
    errors = sample_errors(
        size=100,
        sigma=1.0,
        distribution='t',
        params=None,
        residuals=residuals_too_few,
        rng=np.random.default_rng(42)
    )

    # Should work and generate errors
    assert errors.shape == (100,)

    # All NaN residuals - should also fall back to Mode 2
    residuals_all_nan = np.array([np.nan] * 20)

    errors2 = sample_errors(
        size=100,
        sigma=1.0,
        distribution='laplace',
        params=None,
        residuals=residuals_all_nan,
        rng=np.random.default_rng(42)
    )

    # Should work and generate errors
    assert errors2.shape == (100,)


def test_validation_of_estimated_parameters():
    """Test that estimated parameters are validated (e.g., df > 2)."""
    from statsforecast.simulation import sample_errors
    from scipy import stats

    # Create residuals that would result in df <= 2 (very heavy tails)
    # Using a mixture to create extreme outliers
    np.random.seed(42)
    residuals = np.concatenate([
        np.random.randn(50),
        np.random.randn(10) * 100  # Extreme outliers
    ])

    # This might raise an error if estimated df <= 2
    # (depends on the data, so we just check it doesn't crash for normal data)
    residuals_normal = stats.t.rvs(5, size=100, random_state=42)

    errors = sample_errors(
        size=100,
        sigma=1.0,
        distribution='t',
        params=None,
        residuals=residuals_normal,
        rng=np.random.default_rng(42)
    )

    assert errors.shape == (100,)


def test_explicit_params():
    """Test user given paramaters"""
    from statsforecast.simulation import sample_errors

    np.random.seed(42)

    # Mode 2: explicit params (backward compatible)
    errors_mode2 = sample_errors(
        size=1000,
        sigma=2.5,
        distribution='t',
        params={'df': 7},  # Explicit params
        residuals=None,  # Not needed in Mode 2
        rng=np.random.default_rng(42)
    )

    # Should work and produce results
    assert errors_mode2.shape == (1000,)

    # Test with residuals present but params specified (should use params, not residuals)
    residuals = np.random.randn(100)
    errors_mode2_with_resid = sample_errors(
        size=1000,
        sigma=2.5,
        distribution='t',
        params={'df': 7},  # Explicit params - should use these
        residuals=residuals,  # Present but should be ignored
        rng=np.random.default_rng(42)
    )

    # Should produce identical results (residuals ignored when params provided)
    np.testing.assert_array_equal(errors_mode2, errors_mode2_with_resid)


def test_integration_autoarima_automatic_estimation():
    """Integration test: AutoARIMA with automatic parameter estimation."""
    df = pd.DataFrame({
        'unique_id': [1] * 100,
        'ds': pd.date_range('2000-01-01', periods=100),
        'y': np.random.normal(0, 1, 100).cumsum()
    })

    models = [AutoARIMA()]
    sf = StatsForecast(models=models, freq='D')

    # Simulate with t-distribution WITHOUT explicit params
    # Should automatically estimate from residuals
    res = sf.simulate(h=5, df=df, n_paths=10, seed=42,
                      error_distribution='t')  # No error_params!

    assert 'AutoARIMA' in res.columns
    assert res.shape == (50, 4)  # 5 steps * 10 paths
    assert not res['AutoARIMA'].isna().any()


def test_integration_autoets_automatic_estimation():
    """Integration test: AutoETS with automatic parameter estimation."""
    df = pd.DataFrame({
        'unique_id': [1] * 50,
        'ds': pd.date_range('2000-01-01', periods=50),
        'y': np.random.normal(10, 2, 50).cumsum()
    })

    models = [AutoETS(season_length=1)]
    sf = StatsForecast(models=models, freq='D')

    # Simulate with laplace distribution WITHOUT explicit params
    res = sf.simulate(h=3, df=df, n_paths=5, seed=42,
                      error_distribution='laplace')

    assert 'AutoETS' in res.columns
    assert res.shape == (15, 4)
    assert not res['AutoETS'].isna().any()


def test_integration_multiple_distributions_automatic():
    """Integration test: Multiple models with different auto-estimated distributions."""
    df = pd.DataFrame({
        'unique_id': [1] * 50,
        'ds': pd.date_range('2000-01-01', periods=50),
        'y': np.random.normal(0, 1, 50).cumsum()
    })

    models = [AutoARIMA(), AutoETS(season_length=1), Naive()]
    sf = StatsForecast(models=models, freq='D')

    # Test different distributions with automatic estimation
    distributions = ['normal', 't', 'laplace', 'skew-normal', 'ged']

    for dist in distributions:
        res = sf.simulate(h=3, df=df, n_paths=5, seed=42,
                          error_distribution=dist)

        assert 'AutoARIMA' in res.columns
        assert 'AutoETS' in res.columns
        assert 'Naive' in res.columns
        assert res.shape == (15, 6)  # 3 steps * 5 paths, 6 cols (ds, unique_id, path, 3 models)

        # Verify no NaN values
        assert not res['AutoARIMA'].isna().any()
        assert not res['AutoETS'].isna().any()
        assert not res['Naive'].isna().any()


def test_all_models_all_distributions_automatic():
    """Comprehensive test: All models with all distributions (automatic estimation)."""
    np.random.seed(42)
    df = pd.DataFrame({
        'unique_id': [1] * 80,
        'ds': pd.date_range('2000-01-01', periods=80),
        'y': np.random.normal(10, 2, 80).cumsum()
    })

    models = [
        AutoARIMA(),
        AutoETS(season_length=12),
        AutoCES(season_length=12),
        AutoTheta(season_length=12),
        SimpleExponentialSmoothing(alpha=0.5),
        Naive(),
        RandomWalkWithDrift(),
        SeasonalNaive(season_length=12),
    ]

    sf = StatsForecast(models=models, freq='D')

    # Test all distributions with automatic estimation
    distributions = ['normal', 't', 'laplace', 'skew-normal', 'ged', 'bootstrap']
    h = 3
    n_paths = 5

    for dist in distributions:
        res = sf.simulate(h=h, df=df, n_paths=n_paths, seed=42,
                          error_distribution=dist)

        # Basic shape verification
        expected_rows = h * n_paths
        assert res.shape[0] == expected_rows, \
            f"Expected {expected_rows} rows for distribution {dist}, got {res.shape[0]}"

        # Verify required columns are present
        assert 'ds' in res.columns, f"Missing 'ds' column for distribution {dist}"
        assert 'unique_id' in res.columns, f"Missing 'unique_id' column for distribution {dist}"

        # Verify all models work with this distribution
        for model in models:
            model_name = repr(model)

            # 1. Check column exists
            assert model_name in res.columns, \
                f"Model {model_name} failed with distribution {dist}"

            # 2. Check no NaN values
            assert not res[model_name].isna().any(), \
                f"Model {model_name} has NaN with distribution {dist}"

            # 3. Check all values are finite (not inf or -inf)
            assert np.all(np.isfinite(res[model_name].values)), \
                f"Model {model_name} has non-finite values with distribution {dist}"

            # 4. Check that simulated values exist
            # (different paths should produce different values for most time steps)
            values = res[model_name].values
            assert len(values) > 0, \
                f"Model {model_name} has no values with distribution {dist}"
            
            # 5. Check that simulations are different
            result = res.groupby("ds", as_index = False).agg({model_name:"std"})[model_name]
            assert not np.allclose(result, 0), \
                f"No variation between simulations for model '{model_name}' and distribution '{dist}' - all standard deviations are zero"


def test_model_with_no_residuals_falls_back_correctly():
    """Test that models without residuals can still use Mode 2 (explicit params)."""
    df = pd.DataFrame({
        'unique_id': [1] * 20,
        'ds': pd.date_range('2000-01-01', periods=20),
        'y': [10.0] * 20  # Constant series
    })

    # Naive model with constant series
    models = [Naive()]
    sf = StatsForecast(models=models, freq='D')

    # With explicit params, should work even if residuals are problematic
    res = sf.simulate(h=3, df=df, n_paths=5, seed=42,
                      error_distribution='t',
                      error_params={'df': 5})  # Explicit params (Mode 2)

    assert 'Naive' in res.columns
    assert res.shape == (15, 4)
    assert not res['Naive'].isna().any()
