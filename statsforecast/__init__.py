__version__ = "0.7.0"

from .core import (
    StatsForecast,
    forecast,
    cross_validation,
    ParallelBackend,
    MultiprocessBackend,
    RayBackend,
)
