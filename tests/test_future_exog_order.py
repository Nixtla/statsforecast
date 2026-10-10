import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from statsforecast import StatsForecast
from statsforecast.models import SklearnModel


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("as_polars", [False, True])
@pytest.mark.parametrize("method", ["forecast", "fit_predict", "predict"])
def test_future_exog_column_order(n_jobs, as_polars, method):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 20),
            "ds": np.tile(np.arange(20), 2),
            "first": rng.normal(size=40),
            "second": rng.normal(size=40),
        }
    )
    df["y"] = 2 * df["first"] - 7 * df["second"]
    train = df[df.ds < 17]
    future = df[df.ds >= 17].drop(columns="y")
    # Include an id/time permutation as well as reversing the two features.
    future = future[["second", "ds", "first", "unique_id"]]
    if as_polars:
        train, future = pl.from_pandas(train), pl.from_pandas(future)
    sf = StatsForecast(
        models=[SklearnModel(LinearRegression())], freq=1, n_jobs=n_jobs
    )
    if method == "predict":
        result = sf.fit(train).predict(h=3, X_df=future)
    else:
        result = getattr(sf, method)(df=train, h=3, X_df=future)
    np.testing.assert_allclose(
        result["LinearRegression"].to_numpy(), df.loc[df.ds >= 17, "y"]
    )
