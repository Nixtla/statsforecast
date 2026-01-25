"""
Unobserved Components Model (UCM) wrapper for statsforecast.

This module provides a wrapper around statsmodels.tsa.statespace.structural.UnobservedComponents
to make it compatible with the statsforecast API.

UCM decomposes a time series into:
- Level (intercept that varies over time)
- Trend (slope that varies over time)
- Seasonal (periodic patterns)
- Cycle (longer-term oscillations)
- Irregular (noise)

References:
    - Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter.
    - statsmodels documentation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np


def _calculate_sigma(residuals: np.ndarray, n: int) -> float:
    """Calculate standard error of residuals."""
    return np.sqrt(np.sum(residuals ** 2) / max(n - 1, 1))


def _add_fitted_pi(
        res: Dict[str, Any], se: float, level: List[int]
) -> Dict[str, Any]:
    """Add prediction intervals to fitted values."""
    from scipy.stats import norm

    fitted = res["fitted"]
    for lv in level:
        z = norm.ppf(0.5 + lv / 200)
        res[f"lo-{lv}"] = fitted - z * se
        res[f"hi-{lv}"] = fitted + z * se
    return res


class UCM:
    r"""Unobserved Components Model (UCM).
    
    Also known as Structural Time Series Model. Decomposes a univariate time series 
    into trend, seasonal, cyclical, and irregular components using state space methods
    and the Kalman filter.
    
    This is a wrapper around `statsmodels.tsa.statespace.structural.UnobservedComponents`.
    
    Args:
        level (Union[bool, str], default="local level"): 
            Level component specification. Can be:
            - False: No level component
            - True or "local level": Random walk level
            - "local linear trend" or "lltrend": Level + trend (both stochastic)
            - "random walk with drift" or "rwdrift": Random walk + deterministic drift
            - "smooth trend" or "strend": Smooth trend (integrated random walk)
            - "random trend" or "rtrend": Random trend
            See statsmodels documentation for full list.
        trend (bool, default=False): 
            Whether to include a trend component. Only used if `level` is bool.
        seasonal (Optional[int], default=None): 
            Period of seasonal component. If None, no seasonal component.
        cycle (bool, default=False): 
            Whether to include a cycle component.
        autoregressive (Optional[int], default=None): 
            Order of autoregressive component. If None, no AR component.
        irregular (bool, default=True): 
            Whether to include an irregular (noise) component.
        stochastic_level (bool, default=True): 
            Whether level component is stochastic (if included).
        stochastic_trend (bool, default=True): 
            Whether trend component is stochastic (if included).
        stochastic_seasonal (bool, default=True): 
            Whether seasonal component is stochastic (if included).
        stochastic_cycle (bool, default=False): 
            Whether cycle component is stochastic (if included).
        damped_cycle (bool, default=False): 
            Whether cycle component is damped.
        cycle_period_bounds (Optional[tuple], default=None): 
            Bounds on cycle period (lower, upper). Default is (1.5, 12).
        use_exact_diffuse (bool, default=False): 
            Whether to use exact diffuse initialization.
        fit_method (str, default="lbfgs"): 
            Optimization method for fitting.
        maxiter (int, default=500): 
            Maximum iterations for fitting.
        alias (Optional[str], default=None): 
            Custom name for the model.
    
    Examples:
        >>> from statsforecast.models import UCM
        >>> import numpy as np
        >>> # Local level model (random walk)
        >>> model = UCM(level='local level')
        >>> y = np.cumsum(np.random.randn(100)) + 50
        >>> model.fit(y)
        >>> forecast = model.predict(h=10)
        >>> 
        >>> # Local linear trend with seasonal
        >>> model = UCM(level='local linear trend', seasonal=12)
        >>> model.fit(y)
        >>> forecast = model.predict(h=12, level=[90, 95])

    References:
        - Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter.
        - https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html
    """

    uses_exog = True

    def __init__(
            self,
            level: Union[bool, str] = "local level",
            trend: bool = False,
            seasonal: Optional[int] = None,
            cycle: bool = False,
            autoregressive: Optional[int] = None,
            irregular: bool = True,
            stochastic_level: bool = True,
            stochastic_trend: bool = True,
            stochastic_seasonal: bool = True,
            stochastic_cycle: bool = False,
            damped_cycle: bool = False,
            cycle_period_bounds: Optional[tuple] = None,
            use_exact_diffuse: bool = False,
            fit_method: str = "lbfgs",
            maxiter: int = 500,
            alias: Optional[str] = None,
    ):
        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.cycle = cycle
        self.autoregressive = autoregressive
        self.irregular = irregular
        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        self.stochastic_cycle = stochastic_cycle
        self.damped_cycle = damped_cycle
        self.cycle_period_bounds = cycle_period_bounds
        self.use_exact_diffuse = use_exact_diffuse
        self.fit_method = fit_method
        self.maxiter = maxiter
        self.alias = alias if alias is not None else self._default_alias()

        # Will be set during fit
        self.model_: Optional[Dict[str, Any]] = None

    def _default_alias(self) -> str:
        """Generate default alias based on model specification."""
        parts = ["UCM"]
        if isinstance(self.level, str):
            parts.append(self.level.replace(" ", "_"))
        elif self.level:
            parts.append("level")
            if self.trend:
                parts.append("trend")
        if self.seasonal:
            parts.append(f"s{self.seasonal}")
        if self.cycle:
            parts.append("cycle")
        if self.autoregressive:
            parts.append(f"ar{self.autoregressive}")
        return "_".join(parts)

    def __repr__(self) -> str:
        return self.alias

    def new(self):
        """Create a copy of the model."""
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b

    def _build_model(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """Build statsmodels UCM model."""
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        kwargs = {
            "endog": y,
            "stochastic_seasonal": self.stochastic_seasonal,
            "stochastic_cycle": self.stochastic_cycle,
            "damped_cycle": self.damped_cycle,
            "use_exact_diffuse": self.use_exact_diffuse,
        }

        # Handle level/trend specification
        # When using string specification, don't pass irregular/stochastic params
        # as they may conflict with the string spec defaults
        if isinstance(self.level, str):
            kwargs["level"] = self.level
            # Only pass irregular if it's explicitly False (to override string default)
            if not self.irregular:
                kwargs["irregular"] = False
        else:
            kwargs["level"] = self.level
            kwargs["trend"] = self.trend
            kwargs["irregular"] = self.irregular
            kwargs["stochastic_level"] = self.stochastic_level
            kwargs["stochastic_trend"] = self.stochastic_trend

        # Optional components
        if self.seasonal is not None:
            kwargs["seasonal"] = self.seasonal

        if self.cycle:
            kwargs["cycle"] = True
            if self.cycle_period_bounds is not None:
                kwargs["cycle_period_bounds"] = self.cycle_period_bounds

        if self.autoregressive is not None:
            kwargs["autoregressive"] = self.autoregressive

        if X is not None:
            kwargs["exog"] = X

        return UnobservedComponents(**kwargs)

    def fit(
            self,
            y: np.ndarray,
            X: Optional[np.ndarray] = None,
    ) -> "UCM":
        r"""Fit the UCM model.
        
        Args:
            y (numpy.array): Time series of shape (t,).
            X (numpy.array, optional): Exogenous variables of shape (t, n_x).
        
        Returns:
            UCM: Fitted UCM object.
        """
        y = np.asarray(y, dtype=np.float64)
        if X is not None:
            X = np.asarray(X, dtype=np.float64)

        mod = self._build_model(y, X)

        try:
            res = mod.fit(method=self.fit_method, maxiter=self.maxiter, disp=False)
        except Exception:
            # Fallback to powell optimizer if default fails
            res = mod.fit(method="powell", maxiter=self.maxiter, disp=False)

        # Handle both pandas and numpy returns
        fitted_vals = res.fittedvalues
        if hasattr(fitted_vals, 'values'):
            fitted_vals = fitted_vals.values
        fitted_vals = np.asarray(fitted_vals)

        self.model_ = {
            "model": mod,
            "results": res,
            "fitted": fitted_vals,
            "y": y,
            "X": X,
        }

        # Calculate sigma from residuals
        residuals = y - fitted_vals
        self.model_["sigma"] = _calculate_sigma(residuals, y.size)

        return self

    def predict(
            self,
            h: int,
            X: Optional[np.ndarray] = None,
            level: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        r"""Predict with fitted UCM.
        
        Args:
            h (int): Forecast horizon.
            X (numpy.array, optional): Future exogenous variables of shape (h, n_x).
            level (List[int], optional): Confidence levels (0-100) for prediction intervals.
        
        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for intervals.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        res = self.model_["results"]

        # Get forecast
        if X is not None:
            X = np.asarray(X, dtype=np.float64)
            forecast = res.get_forecast(steps=h, exog=X)
        else:
            forecast = res.get_forecast(steps=h)

        # Handle both numpy array and pandas Series returns
        pred_mean = forecast.predicted_mean
        if hasattr(pred_mean, 'values'):
            pred_mean = pred_mean.values
        result = {"mean": np.asarray(pred_mean)}

        # Add prediction intervals
        if level is not None:
            level = sorted(level)
            for lv in level:
                alpha = 1 - lv / 100
                ci = forecast.conf_int(alpha=alpha)
                # Handle both numpy array and pandas DataFrame returns
                if hasattr(ci, 'iloc'):
                    result[f"lo-{lv}"] = ci.iloc[:, 0].values
                    result[f"hi-{lv}"] = ci.iloc[:, 1].values
                else:
                    result[f"lo-{lv}"] = np.asarray(ci[:, 0])
                    result[f"hi-{lv}"] = np.asarray(ci[:, 1])

        return result

    def predict_in_sample(
            self,
            level: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        r"""Access fitted UCM in-sample predictions.
        
        Args:
            level (List[int], optional): Confidence levels (0-100) for prediction intervals.
        
        Returns:
            dict: Dictionary with entries `fitted` for point predictions.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        result = {"fitted": self.model_["fitted"]}

        if level is not None:
            level = sorted(level)
            result = _add_fitted_pi(
                res=result, se=self.model_["sigma"], level=level
            )

        return result

    def forecast(
            self,
            y: np.ndarray,
            h: int,
            X: Optional[np.ndarray] = None,
            X_future: Optional[np.ndarray] = None,
            level: Optional[List[int]] = None,
            fitted: bool = False,
    ) -> Dict[str, Any]:
        r"""Memory-efficient UCM predictions.
        
        This method avoids memory burden from object storage.
        It is analogous to `fit` + `predict` without storing information.
        
        Args:
            y (numpy.array): Time series of shape (t,).
            h (int): Forecast horizon.
            X (numpy.array, optional): In-sample exogenous variables of shape (t, n_x).
            X_future (numpy.array, optional): Future exogenous variables of shape (h, n_x).
            level (List[int], optional): Confidence levels (0-100) for prediction intervals.
            fitted (bool, default=False): Whether to return in-sample predictions.
        
        Returns:
            dict: Dictionary with entries `mean` for point predictions and optionally `fitted`.
        """
        y = np.asarray(y, dtype=np.float64)
        if X is not None:
            X = np.asarray(X, dtype=np.float64)
        if X_future is not None:
            X_future = np.asarray(X_future, dtype=np.float64)

        mod = self._build_model(y, X)

        try:
            res = mod.fit(method=self.fit_method, maxiter=self.maxiter, disp=False)
        except Exception:
            res = mod.fit(method="powell", maxiter=self.maxiter, disp=False)

        # Get forecast
        if X_future is not None:
            forecast = res.get_forecast(steps=h, exog=X_future)
        else:
            forecast = res.get_forecast(steps=h)

        # Handle both numpy array and pandas Series returns
        pred_mean = forecast.predicted_mean
        if hasattr(pred_mean, 'values'):
            pred_mean = pred_mean.values
        result = {"mean": np.asarray(pred_mean)}

        if fitted:
            fitted_vals = res.fittedvalues
            if hasattr(fitted_vals, 'values'):
                fitted_vals = fitted_vals.values
            result["fitted"] = np.asarray(fitted_vals)

        # Add prediction intervals
        if level is not None:
            level = sorted(level)
            for lv in level:
                alpha = 1 - lv / 100
                ci = forecast.conf_int(alpha=alpha)
                # Handle both numpy array and pandas DataFrame returns
                if hasattr(ci, 'iloc'):
                    result[f"lo-{lv}"] = ci.iloc[:, 0].values
                    result[f"hi-{lv}"] = ci.iloc[:, 1].values
                else:
                    result[f"lo-{lv}"] = np.asarray(ci[:, 0])
                    result[f"hi-{lv}"] = np.asarray(ci[:, 1])

        return result

    def get_components(self) -> Dict[str, np.ndarray]:
        r"""Get decomposed components from fitted model.
        
        Returns:
            dict: Dictionary with available components (level, trend, seasonal, cycle).
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        res = self.model_["results"]
        components = {}

        # Try to extract each component
        if hasattr(res, "level") and res.level is not None:
            components["level"] = res.level.smoothed

        if hasattr(res, "trend") and res.trend is not None:
            components["trend"] = res.trend.smoothed

        if hasattr(res, "seasonal") and res.seasonal is not None:
            components["seasonal"] = res.seasonal.smoothed

        if hasattr(res, "cycle") and res.cycle is not None:
            components["cycle"] = res.cycle.smoothed

        if hasattr(res, "autoregressive") and res.autoregressive is not None:
            components["autoregressive"] = res.autoregressive.smoothed

        return components


# Convenience aliases for common UCM specifications
class LocalLevel(UCM):
    """Local Level model (random walk).
    
    The simplest UCM model: y_t = level_t + eps_t, where level follows a random walk.
    """

    def __init__(
            self,
            seasonal: Optional[int] = None,
            fit_method: str = "lbfgs",
            maxiter: int = 500,
            alias: Optional[str] = None,
    ):
        super().__init__(
            level="local level",
            seasonal=seasonal,
            fit_method=fit_method,
            maxiter=maxiter,
            alias=alias or ("LocalLevel" + (f"_s{seasonal}" if seasonal else "")),
        )


class LocalLinearTrend(UCM):
    """Local Linear Trend model.
    
    Level and trend both follow random walks:
    - level_t = level_{t-1} + trend_{t-1} + eta_t
    - trend_t = trend_{t-1} + zeta_t
    """

    def __init__(
            self,
            seasonal: Optional[int] = None,
            fit_method: str = "lbfgs",
            maxiter: int = 500,
            alias: Optional[str] = None,
    ):
        super().__init__(
            level="local linear trend",
            seasonal=seasonal,
            fit_method=fit_method,
            maxiter=maxiter,
            alias=alias or ("LocalLinearTrend" + (f"_s{seasonal}" if seasonal else "")),
        )


class SmoothTrend(UCM):
    """Smooth Trend model (integrated random walk).
    
    The trend is an integrated random walk, producing smoother forecasts.
    """

    def __init__(
            self,
            seasonal: Optional[int] = None,
            fit_method: str = "lbfgs",
            maxiter: int = 500,
            alias: Optional[str] = None,
    ):
        super().__init__(
            level="smooth trend",
            seasonal=seasonal,
            fit_method=fit_method,
            maxiter=maxiter,
            alias=alias or ("SmoothTrend" + (f"_s{seasonal}" if seasonal else "")),
        )
