import inspect

import numpy as np
import pytest

import statsforecast.models


models = [
    statsforecast.models.AutoARIMA,
    statsforecast.models.AutoCES,
    statsforecast.models.AutoETS,
    statsforecast.models.AutoTheta,
    statsforecast.models.TBATS,
    statsforecast.models.GARCH,
]


@pytest.fixture(scope="module")
def y():
    return np.arange(24)[np.arange(200) % 24]


@pytest.mark.parametrize("model_cls", models)
def test_efficiency(benchmark, y, model_cls):
    if model_cls is statsforecast.models.AutoCES:
        # fails to fit without model="N"
        model = model_cls(season_length=24, model="N")
    elif "season_length" in inspect.signature(model_cls).parameters:
        model = model_cls(season_length=24)
    else:
        model = model_cls()
    benchmark(model.forecast, y=y, h=48)
