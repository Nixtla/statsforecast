"""The short/constant-series fallback must retain the original target scale."""

import numpy as np
import pytest

from statsforecast.mfles import MFLES as BackendMFLES
from statsforecast.models import AutoMFLES, MFLES


CASES = [
    ([7.0] * 10, False),
    ([-3.0] * 10, False),
    ([0.0] * 10, False),
    ([0.1] * 10, False),
    ([7.0] * 10, True),
    ([5.0], False),
    ([5.0, 7.0], False),
    ([-5.0, -7.0, -9.0], False),
    ([5.0], True),
    ([5.0, 7.0], True),
    ([2.0, 5.0, 9.0], True),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("values,multiplicative", CASES)
def test_backend_naive_fallback_original_scale(values, multiplicative, dtype):
    y = np.array(values, dtype=dtype)
    original = y.copy()
    model = BackendMFLES(verbose=False)
    fitted = model.fit(y, multiplicative=multiplicative)
    np.testing.assert_allclose(fitted, y[-1], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model.predict(5), y[-1], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(y, original)


def make_model(cls, multiplicative):
    if cls is MFLES:
        return cls(multiplicative=multiplicative)
    return cls(
        test_size=1,
        n_windows=1,
        config={
            "seasonal_period": [None],
            "max_rounds": [4],
            "multiplicative": [multiplicative],
        },
    )


@pytest.mark.parametrize("cls", [MFLES, AutoMFLES])
@pytest.mark.parametrize("values,multiplicative", CASES)
def test_public_model_naive_fallback_original_scale(cls, values, multiplicative):
    y = np.array(values)
    model = make_model(cls, multiplicative).fit(y)
    np.testing.assert_allclose(model.predict(5)["mean"], y[-1], atol=1e-7)
    np.testing.assert_allclose(model.predict_in_sample()["fitted"], y[-1], atol=1e-7)
    forecast = make_model(cls, multiplicative).forecast(y, h=5, fitted=True)
    np.testing.assert_allclose(forecast["mean"], y[-1], atol=1e-7)
    np.testing.assert_allclose(forecast["fitted"], y[-1], atol=1e-7)


@pytest.mark.parametrize("multiplicative", [False, True])
def test_fallback_replaces_previous_fit_with_exogenous_models(multiplicative):
    model = BackendMFLES(verbose=False)
    t = np.arange(30, dtype=float)
    X = t[:, None]
    model.fit(10 + 0.2 * t + np.sin(t), X=X, max_rounds=4)
    y = np.array([5.0, 7.0])
    fitted = model.fit(y, X=X[:2], multiplicative=multiplicative)
    np.testing.assert_allclose(fitted, 7.0)
    np.testing.assert_allclose(model.predict(4, X=X[:4]), 7.0)
