import numpy as np
import pytest

from statsforecast.models import AutoTheta, DynamicOptimizedTheta, OptimizedTheta
from statsforecast.utils import AirPassengers as ap


@pytest.mark.parametrize(
    "model",
    [
        AutoTheta(season_length=12, model="OTM", theta=2.5),
        OptimizedTheta(season_length=12, theta=2.5),
        DynamicOptimizedTheta(season_length=12, theta=2.5),
    ],
)
def test_fixed_theta_is_used_by_fit_and_forecast(model):
    expected = model.forecast(ap, h=12, level=[80, 95])

    model.fit(ap)
    actual = model.predict(h=12, level=[80, 95])

    assert model.model_["par"]["theta"] == 2.5
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])


@pytest.mark.parametrize("theta", [0.99, np.nan, np.inf, -np.inf, 1e10 + 1])
def test_fixed_theta_validates_optimizer_bounds(theta):
    with pytest.raises(ValueError, match="theta must be finite and between 1 and 1e10"):
        OptimizedTheta(theta=theta)
