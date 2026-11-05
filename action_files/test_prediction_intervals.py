import numpy as np
import pytest
from scipy.stats import norm

from statsforecast.models import Naive


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-25, 1.0, 1e20])
def test_naive_prediction_intervals_preserve_residual_scale(dtype, scale):
    # All three one-step errors have magnitude scale, hence sigma = scale.
    y = np.array([0, scale, 0, scale], dtype=dtype)
    expected_sigma = float(y[1])
    model = Naive().fit(y)
    forecasts = [model.predict(1, level=[95]), model.forecast(y, 1, level=[95])]
    for forecast in forecasts:
        expected_width = norm.ppf(0.975) * expected_sigma
        np.testing.assert_allclose(
            forecast["hi-95"] - forecast["mean"], expected_width, rtol=1e-6, atol=0
        )
        np.testing.assert_allclose(
            forecast["mean"] - forecast["lo-95"], expected_width, rtol=1e-6, atol=0
        )
