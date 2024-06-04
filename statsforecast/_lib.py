import ctypes
import platform
import sys

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


if platform.system() in ("Windows", "Microsoft"):
    _prefix = "Release"
    _extension = "dll"
else:
    _prefix = ""
    _extension = "so"

_LIB = ctypes.CDLL(
    str(files("statsforecast") / "lib" / _prefix / f"libstatsforecast.{_extension}")
)
