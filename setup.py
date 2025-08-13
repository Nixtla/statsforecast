import glob

import setuptools
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, naive_recompile

ext_modules = [
    Pybind11Extension(
        name="statsforecast._lib",
        sources=glob.glob("src/*.cpp"),
        include_dirs=["include/statsforecast", "external_libs/eigen"],
        cxx_std=20,
    )
]

ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL", needs_recompile=naive_recompile).install()

setuptools.setup(ext_modules=ext_modules)
