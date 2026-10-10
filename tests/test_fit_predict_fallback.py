import numpy as np
import pandas as pd
import polars as pl
import pytest

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("as_polars", [False, True])
def test_fit_predict_uses_fallback(n_jobs, as_polars):
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["positive", "zeros"], 12),
            "ds": np.tile(np.arange(12), 2),
            "y": np.r_[np.arange(1.0, 13.0), np.arange(12.0)],
        }
    )
    if as_polars:
        df = pl.from_pandas(df)
    kwargs = dict(models=[AutoETS(model="MNN", alias="ets")], freq=1, n_jobs=n_jobs)
    with pytest.raises(ValueError, match="negative or zero"):
        StatsForecast(**kwargs).fit_predict(df=df, h=3)

    fallback = Naive(alias="fallback")
    sf = StatsForecast(**kwargs, fallback_model=fallback)
    expected = StatsForecast(**kwargs, fallback_model=fallback).fit(df).predict(
        h=3, level=[80]
    )
    result = sf.fit_predict(df=df, h=3, level=[80])
    if as_polars:
        result, expected = result.to_pandas(), expected.to_pandas()
    pd.testing.assert_frame_equal(result, expected)
    np.testing.assert_array_equal(result.loc[result.unique_id == "zeros", "ets"], 11)
    assert isinstance(sf.fitted_[0, 0], AutoETS)
    assert isinstance(sf.fitted_[1, 0], Naive)
    assert sf.fitted_[1, 0].alias == "ets"
    assert fallback.alias == "fallback"
    later = sf.predict(h=3, level=[80])
    if as_polars:
        later = later.to_pandas()
    pd.testing.assert_frame_equal(later, expected)
