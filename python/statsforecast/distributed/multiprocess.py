__all__ = ['MultiprocessBackend']


from typing import Any

from ..core import ParallelBackend, _StatsForecast

# This parent class holds common `forecast` and `cross_validation` methods
# from `core.StatsForecast` to enable the `FugueBackend` and the `RayBackend`.

# This Parent class is inherited by [FugueBakend](https://nixtlaverse.nixtla.io/statsforecast/distributed.fugue)
# and [RayBackend](https://nixtlaverse.nixtla.io/statsforecast/distributed.ray).


class MultiprocessBackend(ParallelBackend):
    """MultiprocessBackend Parent Class for Distributed Computation.

    Args:
        n_jobs (int): Number of jobs used in the parallel processing, use -1 for all cores.
    """

    def __init__(self, n_jobs: int) -> None:
        self.n_jobs = n_jobs
        super().__init__()

    def forecast(self, df, models, freq, fallback_model=None, **kwargs: Any) -> Any:
        model = _StatsForecast(
            models=models, freq=freq, fallback_model=fallback_model, n_jobs=self.n_jobs
        )
        return model.forecast(df=df, **kwargs)

    def cross_validation(
        self, df, models, freq, fallback_model=None, **kwargs: Any
    ) -> Any:
        model = _StatsForecast(
            models=models, freq=freq, fallback_model=fallback_model, n_jobs=self.n_jobs
        )
        return model.cross_validation(df=df, **kwargs)
