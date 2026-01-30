"""Time series data simulation for benchmarking.

This module provides utilities for generating synthetic time series data
with various statistical distributions and custom patterns.
"""

__all__ = ["TimeSeriesSimulator"]

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class TimeSeriesSimulator:
    """Generate synthetic time series with specified distributions.

    The TimeSeriesSimulator generates synthetic time series data that follows
    a specified statistical distribution or custom pattern. This is useful for
    benchmarking forecasting models against known data characteristics.

    Parameters
    ----------
    length : int, default=100
        Length of the time series to generate.
    distribution : str or callable, default="normal"
        Distribution to use for generating the time series values.
        If str, must be one of:

        - "normal" : Normal/Gaussian distribution
        - "poisson" : Poisson distribution
        - "exponential" : Exponential distribution
        - "gamma" : Gamma distribution
        - "uniform" : Uniform distribution
        - "binomial" : Binomial distribution
        - "lognormal" : Log-normal distribution

        If callable, must accept parameters (size, rng) and return array.
        The rng is a numpy.random.Generator for reproducibility.
    dist_params : dict, optional
        Parameters to pass to the distribution function.
        For built-in distributions:

        - "normal": {"loc": 0, "scale": 1}
        - "poisson": {"lam": 5}
        - "exponential": {"scale": 1.0}
        - "gamma": {"shape": 2.0, "scale": 2.0}
        - "uniform": {"low": 0, "high": 1}
        - "binomial": {"n": 10, "p": 0.5}
        - "lognormal": {"mean": 0, "sigma": 1}

    trend : str or callable, optional
        Trend component to add to the series.
        If str, must be one of: "linear", "quadratic", "exponential".
        If callable, must accept array of time indices and return trend values.
    trend_params : dict, optional
        Parameters for the trend function.

        - For "linear": {"slope": 1.0, "intercept": 0.0}
        - For "quadratic": {"a": 0.01, "b": 0, "c": 0}
        - For "exponential": {"base": 1.01, "scale": 1.0}

    seasonality : int or list of int, optional
        Seasonal period(s) to add to the series.
    seasonality_strength : float or list of float, default=1.0
        Amplitude of seasonal component(s).
    noise_std : float, default=0.0
        Standard deviation of additional Gaussian noise.
    freq : str, default="D"
        Frequency of the time index. Pandas frequency string.
    start : str, default="2020-01-01"
        Start date for the time index.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from statsforecast.synthetic import TimeSeriesSimulator

    >>> # Generate normal distributed time series
    >>> sim = TimeSeriesSimulator(
    ...     length=100,
    ...     distribution="normal",
    ...     dist_params={"loc": 100, "scale": 10},
    ...     seed=42,
    ... )
    >>> df = sim.simulate(n_series=5)

    >>> # Generate with custom distribution (demand with spikes)
    >>> def demand_with_spikes(size, rng):
    ...     base_demand = rng.gamma(shape=5, scale=10, size=size)
    ...     spike_mask = rng.random(size) < 0.05  # 5% spike probability
    ...     spike_multiplier = rng.uniform(2.5, 5.0, size=size)
    ...     demand = base_demand.copy()
    ...     demand[spike_mask] *= spike_multiplier[spike_mask]
    ...     return demand
    ...
    >>> sim = TimeSeriesSimulator(
    ...     length=300,
    ...     distribution=demand_with_spikes,
    ...     trend="linear",
    ...     trend_params={"slope": 0.02},
    ...     seasonality=7,
    ...     seasonality_strength=5.0,
    ...     seed=42,
    ... )
    >>> df = sim.simulate(n_series=10)

    >>> # Generate with multiple seasonalities
    >>> sim = TimeSeriesSimulator(
    ...     length=365,
    ...     distribution="gamma",
    ...     dist_params={"shape": 2, "scale": 50},
    ...     seasonality=[7, 30],  # weekly and monthly
    ...     seasonality_strength=[10.0, 5.0],
    ...     seed=42,
    ... )
    >>> df = sim.simulate()
    """

    # Built-in distribution names
    _DISTRIBUTIONS = {
        "normal",
        "poisson",
        "exponential",
        "gamma",
        "uniform",
        "binomial",
        "lognormal",
    }

    def __init__(
        self,
        length: int = 100,
        distribution: Union[str, Callable] = "normal",
        dist_params: Optional[Dict] = None,
        trend: Optional[Union[str, Callable]] = None,
        trend_params: Optional[Dict] = None,
        seasonality: Optional[Union[int, List[int]]] = None,
        seasonality_strength: Union[float, List[float]] = 1.0,
        noise_std: float = 0.0,
        freq: str = "D",
        start: str = "2020-01-01",
        seed: Optional[int] = None,
    ):
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")

        # Adjust length for seasonality if needed
        if seasonality is not None:
            if isinstance(seasonality, (list, tuple)):
                max_seasonality = max(seasonality)
            else:
                max_seasonality = seasonality
            min_length = 3 * max_seasonality
            if length < min_length:
                length = min_length

        self.length = length
        self.distribution = distribution
        self.dist_params = dist_params or {}
        self.trend = trend
        self.trend_params = trend_params or {}
        self.seasonality = seasonality
        self.seasonality_strength = seasonality_strength
        self.noise_std = noise_std
        self.freq = freq
        self.start = start
        self.seed = seed

    def simulate(self, n_series: int = 1) -> pd.DataFrame:
        """Generate synthetic time series.

        Parameters
        ----------
        n_series : int, default=1
            Number of series to generate.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: unique_id, ds, y
            Compatible with statsforecast format.
        """
        rng = np.random.default_rng(self.seed)

        # Generate time index
        dates = pd.date_range(start=self.start, periods=self.length, freq=self.freq)

        records = []
        for i in range(n_series):
            values = self._generate_from_distribution(rng)

            if self.trend is not None:
                values = values + self._generate_trend()

            if self.seasonality is not None:
                values = values + self._generate_seasonality()

            if self.noise_std > 0:
                values = values + rng.normal(0, self.noise_std, size=self.length)

            for date, val in zip(dates, values):
                records.append({"unique_id": i, "ds": date, "y": val})

        return pd.DataFrame(records)

    def _generate_from_distribution(self, rng: np.random.Generator) -> np.ndarray:
        """Generate values from the specified distribution."""
        params = self.dist_params

        if callable(self.distribution):
            # Custom distribution function: (size, rng) -> array
            return self.distribution(self.length, rng)

        elif self.distribution == "normal":
            loc = params.get("loc", 0)
            scale = params.get("scale", 1)
            return rng.normal(loc, scale, size=self.length)

        elif self.distribution == "poisson":
            lam = params.get("lam", 5)
            return rng.poisson(lam, size=self.length).astype(float)

        elif self.distribution == "exponential":
            scale = params.get("scale", 1.0)
            return rng.exponential(scale, size=self.length)

        elif self.distribution == "gamma":
            shape = params.get("shape", 2.0)
            scale = params.get("scale", 2.0)
            return rng.gamma(shape, scale, size=self.length)

        elif self.distribution == "uniform":
            low = params.get("low", 0)
            high = params.get("high", 1)
            return rng.uniform(low, high, size=self.length)

        elif self.distribution == "binomial":
            n = params.get("n", 10)
            p = params.get("p", 0.5)
            return rng.binomial(n, p, size=self.length).astype(float)

        elif self.distribution == "lognormal":
            mean = params.get("mean", 0)
            sigma = params.get("sigma", 1)
            return rng.lognormal(mean, sigma, size=self.length)

        else:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. "
                f"Expected one of {sorted(self._DISTRIBUTIONS)} or a callable."
            )

    def _generate_trend(self) -> np.ndarray:
        """Generate trend component."""
        t = np.arange(self.length)
        params = self.trend_params

        if callable(self.trend):
            return self.trend(t)

        elif self.trend == "linear":
            slope = params.get("slope", 1.0)
            intercept = params.get("intercept", 0.0)
            return slope * t + intercept

        elif self.trend == "quadratic":
            a = params.get("a", 0.01)
            b = params.get("b", 0)
            c = params.get("c", 0)
            return a * t**2 + b * t + c

        elif self.trend == "exponential":
            base = params.get("base", 1.01)
            scale = params.get("scale", 1.0)
            return scale * (base**t - 1)

        else:
            raise ValueError(
                f"Unknown trend: {self.trend}. "
                "Must be 'linear', 'quadratic', 'exponential', or a callable."
            )

    def _generate_seasonality(self) -> np.ndarray:
        """Generate seasonal component(s)."""
        t = np.arange(self.length)
        seasonal = np.zeros(self.length)

        # Handle single or multiple seasonal periods
        periods = (
            [self.seasonality]
            if isinstance(self.seasonality, int)
            else list(self.seasonality)
        )
        strengths = (
            [self.seasonality_strength]
            if isinstance(self.seasonality_strength, (int, float))
            else list(self.seasonality_strength)
        )

        if len(strengths) == 1 and len(periods) > 1:
            strengths = strengths * len(periods)
        elif len(strengths) != len(periods):
            raise ValueError(
                "seasonality_strength must be a single value or match "
                "the number of seasonal periods."
            )

        for period, strength in zip(periods, strengths):
            seasonal += strength * np.sin(2 * np.pi * t / period)

        return seasonal
