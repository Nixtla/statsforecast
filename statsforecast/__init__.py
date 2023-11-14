__version__ = "1.6.0"
__all__ = ["StatsForecast"]
from .config import config
from .core import StatsForecast
from .distributed import fugue  # noqa
