"""
Simulation utilities for StatsForecast.

This module provides error distribution sampling functions for Monte Carlo
simulation of forecast paths. Supports multiple error distributions beyond
the standard normal distribution.
"""

__all__ = [
    "sample_errors",
    "SUPPORTED_DISTRIBUTIONS",
]

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats


SUPPORTED_DISTRIBUTIONS = frozenset([
    "normal",
    "t",
    "bootstrap",
    "laplace",
    "skew-normal",
    "ged",
])


def _validate_distribution(distribution: str) -> None:
    """Validate that the distribution is supported."""
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported error distribution: '{distribution}'. "
            f"Supported distributions are: {sorted(SUPPORTED_DISTRIBUTIONS)}"
        )


def sample_errors(
    size: Union[int, Tuple[int, ...]],
    sigma: float,
    distribution: str = "normal",
    params: Optional[Dict] = None,
    residuals: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample errors from a specified distribution for simulation.

    Parameters
    ----------
    size : int or tuple of ints
        Output shape. If (n_paths, h), generates a matrix of errors.
    sigma : float
        Scale parameter (standard deviation for normal-like distributions).
    distribution : str, default='normal'
        The error distribution to use. Options:
        - 'normal': Standard normal distribution (default)
        - 't': Student's t-distribution (requires 'df' in params)
        - 'bootstrap': Resample from empirical residuals (requires residuals)
        - 'laplace': Laplace (double exponential) distribution
        - 'skew-normal': Skewed normal distribution (optional 'skewness' in params)
        - 'ged': Generalized Error Distribution (optional 'shape' in params)
    params : dict, optional
        Distribution-specific parameters:
        - For 't': {'df': degrees_of_freedom} (default: 5)
        - For 'skew-normal': {'skewness': alpha} (default: 0)
        - For 'ged': {'shape': beta} (default: 2, which equals normal)
    residuals : np.ndarray, optional
        Fitted residuals for bootstrap resampling. Required when distribution='bootstrap'.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, uses numpy's default.

    Returns
    -------
    np.ndarray
        Array of sampled errors with the specified shape.

    Examples
    --------
    >>> # Normal errors
    >>> errors = sample_errors(size=(100, 10), sigma=1.0)
    
    >>> # Student's t with 5 degrees of freedom
    >>> errors = sample_errors(size=(100, 10), sigma=1.0, distribution='t', params={'df': 5})
    
    >>> # Bootstrap from residuals
    >>> errors = sample_errors(size=(100, 10), sigma=1.0, distribution='bootstrap', residuals=resid)
    """
    _validate_distribution(distribution)
    
    if params is None:
        params = {}
    
    if rng is None:
        rng = np.random.default_rng()
    
    if distribution == "normal":
        return rng.normal(0, sigma, size)
    
    elif distribution == "t":
        df = params.get("df", 5)
        if df <= 2:
            raise ValueError("Degrees of freedom (df) must be > 2 for finite variance.")
        # Scale t-distribution to have the desired standard deviation
        # Var(t) = df / (df - 2) for df > 2
        t_scale = sigma * np.sqrt((df - 2) / df)
        return stats.t.rvs(df, scale=t_scale, size=size, random_state=rng)
    
    elif distribution == "bootstrap":
        if residuals is None:
            raise ValueError(
                "Bootstrap distribution requires 'residuals' parameter. "
                "Pass the fitted model residuals."
            )
        # Remove NaN values from residuals
        clean_residuals = residuals[~np.isnan(residuals)]
        if len(clean_residuals) == 0:
            raise ValueError("No valid residuals available for bootstrap sampling.")
        # Resample from empirical residuals
        if isinstance(size, int):
            total_size = size
        else:
            total_size = np.prod(size)
        sampled = rng.choice(clean_residuals, size=total_size, replace=True)
        return sampled.reshape(size)
    
    elif distribution == "laplace":
        # Laplace scale parameter: sigma = scale * sqrt(2)
        laplace_scale = sigma / np.sqrt(2)
        return rng.laplace(0, laplace_scale, size)
    
    elif distribution == "skew-normal":
        skewness = params.get("skewness", 0)
        # Sample from skew-normal and scale to desired sigma
        # scipy's skewnorm uses 'a' as shape parameter (skewness)
        raw_samples = stats.skewnorm.rvs(skewness, size=size, random_state=rng)
        # Standardize and rescale
        if np.std(raw_samples) > 0:
            return raw_samples * sigma / np.std(raw_samples)
        return raw_samples * sigma
    
    elif distribution == "ged":
        # Generalized Error Distribution (generalized normal)
        # shape=2 -> normal, shape=1 -> Laplace, shape->inf -> uniform
        shape = params.get("shape", 2)
        if shape <= 0:
            raise ValueError("GED shape parameter must be positive.")
        # scipy's gennorm uses beta as the shape parameter
        raw_samples = stats.gennorm.rvs(shape, size=size, random_state=rng)
        # Scale to desired sigma
        ged_var = stats.gennorm.var(shape)
        if ged_var > 0:
            return raw_samples * sigma / np.sqrt(ged_var)
        return raw_samples * sigma
    
    else:
        # This should not be reached due to validation, but just in case
        raise ValueError(f"Unknown distribution: {distribution}")


def get_distribution_info(distribution: str) -> Dict:
    """
    Get information about a supported distribution.

    Parameters
    ----------
    distribution : str
        Name of the distribution.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'name': Full name of the distribution
        - 'params': List of supported parameters
        - 'description': Brief description
    """
    _validate_distribution(distribution)
    
    info = {
        "normal": {
            "name": "Normal (Gaussian)",
            "params": [],
            "description": "Standard normal distribution. Default choice for most applications.",
        },
        "t": {
            "name": "Student's t",
            "params": ["df (degrees of freedom, default=5)"],
            "description": "Heavy-tailed distribution ideal for financial time series.",
        },
        "bootstrap": {
            "name": "Bootstrap (Empirical)",
            "params": [],
            "description": "Non-parametric resampling from fitted residuals.",
        },
        "laplace": {
            "name": "Laplace (Double Exponential)",
            "params": [],
            "description": "Sharper peak with heavier tails than normal. Good for intermittent demand.",
        },
        "skew-normal": {
            "name": "Skewed Normal",
            "params": ["skewness (alpha, default=0)"],
            "description": "Asymmetric normal distribution. Use for skewed forecast errors.",
        },
        "ged": {
            "name": "Generalized Error Distribution",
            "params": ["shape (beta, default=2)"],
            "description": "Flexible distribution: shape=2 is normal, shape=1 is Laplace.",
        },
    }
    
    return info[distribution]
