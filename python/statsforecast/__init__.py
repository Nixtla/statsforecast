from importlib.metadata import version

__version__ = version("statsforecast")
__all__ = ["StatsForecast"]
from .core import StatsForecast
from .distributed import fugue  # noqa
