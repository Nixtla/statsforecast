__version__ = "2.0.3"
__all__ = ["StatsForecast", "Distribution", "Gaussian", "Poisson", "NegBin", "StudentT"]
from .core import StatsForecast
from .distributed import fugue  # noqa
from .distributions import Distribution, Gaussian, Poisson, NegBin, StudentT
