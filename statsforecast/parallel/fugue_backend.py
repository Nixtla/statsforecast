from typing import Any, Dict

import pandas as pd
from fugue import transform
from statsforecast.core import (
    ParallelBackend,
    StatsForecast,
    _as_tuple,
    _build_forecast_name,
)
from triad import Schema


class FugueBackend(ParallelBackend):
    def __init__(self, engine: Any = None, conf: Any = None, **transform_kwargs: Any):
        self._engine = engine
        self._conf = conf
        self._transform_kwargs = dict(transform_kwargs)

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def forecast(self, df, models, freq, **kwargs: Any) -> Any:
        schema = "*-y+" + str(self._get_output_schema(models))
        return transform(
            df,
            self._forecast_series,
            params=dict(models=models, freq=freq, kwargs=kwargs),
            schema=schema,
            partition={"by": "unique_id"},
            engine=self._engine,
            engine_conf=self._conf,
            **self._transform_kwargs,
        )

    def cross_validation(self, df, models, freq, **kwargs: Any) -> Any:
        schema = "*," + str(self._get_output_schema(models, mode="cv"))
        return transform(
            df,
            self._cv,
            params=dict(models=models, freq=freq, kwargs=kwargs),
            schema=schema,
            partition={"by": "unique_id"},
            engine=self._engine,
            engine_conf=self._conf,
            **self._transform_kwargs,
        )

    def _forecast_series(self, df: pd.DataFrame, models, freq, kwargs) -> pd.DataFrame:
        tdf = df.set_index("unique_id")
        model = StatsForecast(tdf, models=models, freq=freq, n_jobs=1)
        return model.forecast(**kwargs).reset_index()

    def _cv(self, df: pd.DataFrame, models, freq, kwargs) -> pd.DataFrame:
        tdf = df.set_index("unique_id")
        model = StatsForecast(tdf, models=models, freq=freq, n_jobs=1)
        return model.cross_validation(**kwargs).reset_index()

    def _get_output_schema(self, models, mode="forecast") -> Schema:
        cols = []
        for model_args in models:
            model, *args = _as_tuple(model_args)
            cols.append((_build_forecast_name(model, *args), float))
        if mode == "cv":
            cols = [("cutoff", "datetime")] + cols
        return Schema(cols)
