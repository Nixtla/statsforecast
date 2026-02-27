__all__ = ["Distribution", "Gaussian", "Poisson", "NegBin", "StudentT"]

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
from scipy import stats


def _get_distribution(dist: Union[str, "Distribution"]) -> "Distribution":
    if isinstance(dist, Distribution):
        return dist
    mapping = {
        "gaussian": Gaussian(),
        "normal": Gaussian(),
        "poisson": Poisson(),
        "negbin": NegBin(),
        "negative_binomial": NegBin(),
        "student_t": StudentT(),
        "t": StudentT(),
    }
    key = dist.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown distribution '{dist}'. "
            f"Choose from: {list(mapping)}"
        )
    return mapping[key]


class Distribution(ABC):
    name: str

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply link function — override for count distributions."""
        return y

    def inverse_transform(self, y_hat: np.ndarray) -> np.ndarray:
        """Inverse link function."""
        return y_hat

    def validate(self, y: np.ndarray) -> None:
        """Raise ValueError if data is incompatible with this distribution."""
        pass

    def estimate_params(self, residuals: np.ndarray) -> Dict:
        """Estimate auxiliary params from in-sample residuals (original scale)."""
        return {}

    @abstractmethod
    def predict_intervals(
        self,
        mean: np.ndarray,
        sigma: float,
        level: List[int],
        **kw,
    ) -> Dict[str, np.ndarray]:
        """Return dict with keys 'lo-{l}' and 'hi-{l}' for each l in level."""
        ...


class Gaussian(Distribution):
    name = "gaussian"

    def predict_intervals(self, mean, sigma, level, **kw):
        res = {}
        for l in sorted(level):
            z = stats.norm.ppf((100 + l) / 200)
            res[f"lo-{l}"] = mean - z * sigma
            res[f"hi-{l}"] = mean + z * sigma
        return res


class Poisson(Distribution):
    name = "poisson"

    def transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y_hat):
        return np.expm1(y_hat)

    def validate(self, y):
        if np.any(y < 0):
            raise ValueError("Poisson distribution requires non-negative data.")

    def predict_intervals(self, mean, sigma, level, **kw):
        mu = np.maximum(mean, 1e-8)
        return {
            **{
                f"lo-{l}": stats.poisson.ppf((100 - l) / 200, mu=mu)
                for l in sorted(level)
            },
            **{
                f"hi-{l}": stats.poisson.ppf((100 + l) / 200, mu=mu)
                for l in sorted(level)
            },
        }


class NegBin(Distribution):
    name = "negbin"

    def transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y_hat):
        return np.expm1(y_hat)

    def validate(self, y):
        if np.any(y < 0):
            raise ValueError(
                "Negative Binomial distribution requires non-negative data."
            )

    def estimate_params(self, residuals):
        # Method of moments: r = mu² / (var - mu)
        mu = max(float(np.abs(np.mean(residuals))), 1e-8)
        var = max(float(np.var(residuals)), 1e-8)
        r = mu**2 / (var - mu) if var > mu else 1.0
        return {"r": max(r, 0.1)}

    def predict_intervals(self, mean, sigma, level, r=1.0, **kw):
        mu = np.maximum(mean, 1e-8)
        p = r / (r + mu)
        return {
            **{
                f"lo-{l}": stats.nbinom.ppf((100 - l) / 200, n=r, p=p)
                for l in sorted(level)
            },
            **{
                f"hi-{l}": stats.nbinom.ppf((100 + l) / 200, n=r, p=p)
                for l in sorted(level)
            },
        }


class StudentT(Distribution):
    name = "student_t"

    def estimate_params(self, residuals):
        try:
            df, _, scale = stats.t.fit(residuals, floc=0)
            df = max(df, 2.1)
        except Exception:
            df, scale = 5.0, float(np.std(residuals))
        return {"df": float(df), "scale": float(scale)}

    def predict_intervals(self, mean, sigma, level, df=5.0, scale=None, **kw):
        s = scale if scale is not None else sigma
        return {
            **{
                f"lo-{l}": mean - stats.t.ppf((100 + l) / 200, df=df) * s
                for l in sorted(level)
            },
            **{
                f"hi-{l}": mean + stats.t.ppf((100 + l) / 200, df=df) * s
                for l in sorted(level)
            },
        }
