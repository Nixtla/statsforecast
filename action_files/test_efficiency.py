import inspect

import numpy as np
import pytest

import statsforecast.models


models = [
    # Auto models
    statsforecast.models.AutoARIMA,
    statsforecast.models.AutoCES,
    statsforecast.models.AutoETS,
    statsforecast.models.AutoTheta,
    statsforecast.models.AutoMFLES,
    statsforecast.models.AutoTBATS,
    # ARIMA family
    statsforecast.models.ARIMA,
    statsforecast.models.AutoRegressive,
    # Exponential Smoothing
    statsforecast.models.SimpleExponentialSmoothing,
    statsforecast.models.SimpleExponentialSmoothingOptimized,
    statsforecast.models.SeasonalExponentialSmoothing,
    statsforecast.models.SeasonalExponentialSmoothingOptimized,
    statsforecast.models.Holt,
    statsforecast.models.HoltWinters,
    # Baseline
    statsforecast.models.HistoricAverage,
    statsforecast.models.Naive,
    statsforecast.models.RandomWalkWithDrift,
    statsforecast.models.SeasonalNaive,
    # Window
    statsforecast.models.WindowAverage,
    statsforecast.models.SeasonalWindowAverage,
    # Intermittent Demand
    statsforecast.models.ADIDA,
    statsforecast.models.CrostonClassic,
    statsforecast.models.CrostonOptimized,
    statsforecast.models.CrostonSBA,
    statsforecast.models.IMAPA,
    statsforecast.models.TSB,
    # Decomposition
    statsforecast.models.MSTL,
    statsforecast.models.MFLES,
    # Theta family
    statsforecast.models.TBATS,
    statsforecast.models.Theta,
    statsforecast.models.OptimizedTheta,
    statsforecast.models.DynamicTheta,
    statsforecast.models.DynamicOptimizedTheta,
    # Volatility
    statsforecast.models.GARCH,
    statsforecast.models.ARCH,
    # Utility
    statsforecast.models.ConstantModel,
    statsforecast.models.ZeroModel,
    statsforecast.models.NaNModel,
]

_extra_kwargs = {
    statsforecast.models.AutoCES: {"model": "N"},
    statsforecast.models.AutoMFLES: {"test_size": 48},
    statsforecast.models.AutoRegressive: {"lags": [1, 2, 3, 24]},
    statsforecast.models.SimpleExponentialSmoothing: {"alpha": 0.1},
    statsforecast.models.SeasonalExponentialSmoothing: {"alpha": 0.1},
    statsforecast.models.WindowAverage: {"window_size": 4},
    statsforecast.models.SeasonalWindowAverage: {"window_size": 4},
    statsforecast.models.TSB: {"alpha_d": 0.1, "alpha_p": 0.1},
    statsforecast.models.ConstantModel: {"constant": 0.0},
}


@pytest.fixture(scope="module")
def y():
    return np.arange(24)[np.arange(200) % 24]


@pytest.mark.parametrize("model_cls", models)
def test_efficiency(benchmark, y, model_cls):
    kwargs = {}
    if "season_length" in inspect.signature(model_cls).parameters:
        kwargs["season_length"] = 24
    kwargs.update(_extra_kwargs.get(model_cls, {}))
    model = model_cls(**kwargs)
    benchmark(model.forecast, y=y, h=48)
