import numpy as np
import pandas as pd
import pytest

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive


@pytest.mark.parametrize("n_jobs", [1, 2, 3])
@pytest.mark.parametrize("fallback_first", [False, True])
@pytest.mark.parametrize("method", ["predict", "forecast", "cv", "cv_forward"])
def test_fallback_intervals_match_named_model_results(n_jobs, fallback_first, method):
    positive = 10.0 + np.sin(np.arange(30))
    with_zero = positive.copy()
    with_zero[0] = 0
    series = [with_zero, positive] if fallback_first else [positive, with_zero]
    # Include both mixed-model workers and workers containing one model type.
    series *= 2
    df = pd.concat(
        [
            pd.DataFrame({"unique_id": i, "ds": np.arange(y.size), "y": y})
            for i, y in enumerate(series)
        ],
        ignore_index=True,
    )
    h, level = 2, [80, 95]
    sf = StatsForecast(
        models=[AutoETS(model="MNN", alias="model")],
        fallback_model=Naive(),
        freq=1,
        n_jobs=n_jobs,
    )
    if method == "predict":
        result = sf.fit(df).predict(h=h, level=level)
    elif method == "forecast":
        result = sf.forecast(df=df, h=h, level=level, fitted=True)
    else:
        result = sf.cross_validation(
            df=df, h=h, n_windows=2, level=level, refit=method == "cv"
        )
    for i, y in enumerate(series):
        rows = result.loc[result.unique_id == i]
        windows = rows.groupby("cutoff") if method.startswith("cv") else [(None, rows)]
        native = Naive() if (y == 0).any() else AutoETS(model="MNN")
        for cutoff, window in windows:
            train = y if cutoff is None else y[: cutoff + 1]
            if method == "predict":
                expected = native.fit(train).predict(h=h, level=level)
            elif method == "cv_forward":
                if not hasattr(native, "model_"):
                    native.fit(train)
                expected = native.forward(y=train, h=h, level=level)
            else:
                expected = native.forecast(y=train, h=h, level=level)
            for key, values in expected.items():
                column = "model" if key == "mean" else f"model-{key}"
                np.testing.assert_allclose(window[column], values)


def test_fallback_interval_order_can_change_between_cv_windows():
    y = 10.0 + np.sin(np.arange(40))
    y[36] = 0
    df = pd.DataFrame({"unique_id": "series", "ds": np.arange(y.size), "y": y})
    sf = StatsForecast(
        models=[AutoETS(model="MNN")], fallback_model=Naive(), freq=1
    )
    result = sf.cross_validation(
        df=df, h=2, n_windows=3, input_size=12, level=[80, 95]
    )
    model_types = set()
    for cutoff, window in result.groupby("cutoff"):
        train = y[cutoff - 11 : cutoff + 1]
        native = Naive() if (train == 0).any() else AutoETS(model="MNN")
        model_types.add(type(native))
        expected = native.forecast(y=train, h=2, level=[80, 95])
        for key, values in expected.items():
            column = "AutoETS" if key == "mean" else f"AutoETS-{key}"
            np.testing.assert_allclose(window[column], values)
    assert model_types == {AutoETS, Naive}
