__version__ = "0.7.1"

from .core import (
    StatsForecast,
    forecast,
    cross_validation,
    ParallelBackend,
    MultiprocessBackend,
    RayBackend,
)
