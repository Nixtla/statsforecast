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


def _fit_t_distribution(residuals: np.ndarray) -> Tuple[float, float, float]:
    """Fit t-distribution to residuals, return (df, loc, scale)."""
    clean_residuals = residuals[~np.isnan(residuals)]
    if len(clean_residuals) < 10:
        raise ValueError("Need at least 10 residuals for t-distribution estimation")

    # Use MLE via scipy.stats.t.fit()
    df_est, loc_est, scale_est = stats.t.fit(clean_residuals)

    # Validate: df must be > 2 for finite variance
    if df_est <= 2:
        raise ValueError(f"Estimated df={df_est:.2f} is too low (must be > 2)")

    return float(df_est), float(loc_est), float(scale_est)


def _fit_skewnorm_distribution(residuals: np.ndarray) -> Tuple[float, float, float]:
    """Fit skew-normal distribution to residuals, return (skewness, loc, scale)."""
    clean_residuals = residuals[~np.isnan(residuals)]
    if len(clean_residuals) < 10:
        raise ValueError("Need at least 10 residuals for skew-normal estimation")

    # Use MLE via scipy.stats.skewnorm.fit()
    # Returns (a, loc, scale) where a is the skewness parameter
    skewness_est, loc_est, scale_est = stats.skewnorm.fit(clean_residuals)

    return float(skewness_est), float(loc_est), float(scale_est)


def _fit_gennorm_distribution(residuals: np.ndarray) -> Tuple[float, float, float]:
    """Fit Generalized Error Distribution to residuals, return (shape, loc, scale)."""
    clean_residuals = residuals[~np.isnan(residuals)]
    if len(clean_residuals) < 10:
        raise ValueError("Need at least 10 residuals for GED estimation")

    # Use MLE via scipy.stats.gennorm.fit()
    # Returns (beta, loc, scale) where beta is the shape parameter
    shape_est, loc_est, scale_est = stats.gennorm.fit(clean_residuals)

    # Validate: shape must be positive
    if shape_est <= 0:
        raise ValueError(f"Estimated shape={shape_est:.2f} is invalid (must be > 0)")

    return float(shape_est), float(loc_est), float(scale_est)


def _fit_laplace_distribution(residuals: np.ndarray) -> Tuple[float, float]:
    """Fit Laplace distribution to residuals, return (loc, scale)."""
    clean_residuals = residuals[~np.isnan(residuals)]
    if len(clean_residuals) < 10:
        raise ValueError("Need at least 10 residuals for Laplace estimation")

    # Use MLE via scipy.stats.laplace.fit()
    loc_est, scale_est = stats.laplace.fit(clean_residuals)

    return float(loc_est), float(scale_est)


def _fit_normal_distribution(residuals: np.ndarray) -> Tuple[float, float]:
    """Fit normal distribution to residuals, return (loc, scale)."""
    clean_residuals = residuals[~np.isnan(residuals)]
    if len(clean_residuals) < 10:
        raise ValueError("Need at least 10 residuals for normal estimation")

    # Use MLE via scipy.stats.norm.fit()
    loc_est, scale_est = stats.norm.fit(clean_residuals)

    return float(loc_est), float(scale_est)


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
        Scale parameter (standard deviation). Used for 'normal' distribution and
        when explicit params are provided.
    distribution : str, default='normal'
        The error distribution to use. When params=None and residuals are provided,
        all distribution parameters are automatically estimated. Options:
        - 'normal': Standard normal distribution (default)
        - 't': Student's t-distribution
        - 'bootstrap': Resample from empirical residuals (requires residuals)
        - 'laplace': Laplace (double exponential) distribution
        - 'skew-normal': Skewed normal distribution
        - 'ged': Generalized Error Distribution
    params : dict, optional
        Distribution-specific parameters. If None, then it will be estimated with MLE.
        - Uses provided params + sigma
        - For 't': {'df': degrees_of_freedom} (default: 5)
        - For 'skew-normal': {'skewness': alpha} (default: 0)
        - For 'ged': {'shape': beta} (default: 2, which equals normal)

    residuals : np.ndarray, optional
        Fitted residuals for automatic parameter estimation and bootstrap.
        Required when params=None for non-normal distributions, or when
        distribution='bootstrap'.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, uses numpy's default.

    Returns
    -------
    np.ndarray
        Array of sampled errors with the specified shape.

    Raises
    ------
    ValueError
        - If distribution is not supported
        - If residuals are insufficient (<10 samples) for automatic estimation
        - If estimated parameters are invalid (e.g., df <= 2 for t-distribution)
        - If residuals are None when required for Mode 1 or bootstrap

    """
    _validate_distribution(distribution)

    if rng is None:
        rng = np.random.default_rng()

    # Automatic parameter estimation from residuals
    # When residuals are provided and params is None, fit all distribution parameters
    # Check if we have sufficient residuals for automatic estimation
    _has_sufficient_residuals = False
    if residuals is not None:
        clean_residuals = residuals[~np.isnan(residuals)]
        _has_sufficient_residuals = len(clean_residuals) >= 10

    if residuals is not None and params is None and _has_sufficient_residuals:
        if distribution == "normal":
            # Special case: normal uses pre-calculated sigma from model
            # This is already fitted and adjusted for degrees of freedom
            return rng.normal(0, sigma, size)

        elif distribution == "t":
            df_fit, loc_fit, scale_fit = _fit_t_distribution(residuals)
            return stats.t.rvs(df_fit, loc=loc_fit, scale=scale_fit, size=size, random_state=rng)

        elif distribution == "laplace":
            loc_fit, scale_fit = _fit_laplace_distribution(residuals)
            return rng.laplace(loc_fit, scale_fit, size)

        elif distribution == "skew-normal":
            skewness_fit, loc_fit, scale_fit = _fit_skewnorm_distribution(residuals)
            return stats.skewnorm.rvs(skewness_fit, loc=loc_fit, scale=scale_fit, size=size, random_state=rng)

        elif distribution == "ged":
            shape_fit, loc_fit, scale_fit = _fit_gennorm_distribution(residuals)
            return stats.gennorm.rvs(shape_fit, loc=loc_fit, scale=scale_fit, size=size, random_state=rng)

        elif distribution == "bootstrap":
            # Bootstrap always uses residuals directly
            clean_residuals = residuals[~np.isnan(residuals)]
            if len(clean_residuals) == 0:
                raise ValueError("No valid residuals available for bootstrap sampling.")
            if isinstance(size, int):
                total_size = size
            else:
                total_size = np.prod(size)
            sampled = rng.choice(clean_residuals, size=total_size, replace=True)
            return sampled.reshape(size)

    # User provided parameters
    if params is None:
        params = {}

    # Validate that required params are provided when residuals not available
    if residuals is None:
        if distribution in ["t", "skew-normal", "ged"] and not params:
            raise ValueError(
                f"{distribution} distribution requires either 'params' or 'residuals' for parameter estimation. "
                f"Provide params dict or residuals array."
            )
        elif distribution == "bootstrap":
            raise ValueError(
                "Bootstrap distribution requires 'residuals' parameter. "
                "Pass the fitted model residuals."
            )

    # Distribution-specific sampling with explicit params + sigma
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
        # Bootstrap requires residuals (already validated above at lines 222-226)
        clean_residuals = residuals[~np.isnan(residuals)]  # type: ignore[index]
        if len(clean_residuals) == 0:
            raise ValueError("No valid residuals available for bootstrap sampling.")
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
