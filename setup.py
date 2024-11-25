import glob

import setuptools
from configparser import ConfigParser
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, naive_recompile

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for o in expected: assert o in cfg, "missing expected setting: {}".format(o)
setup_cfg = {o:cfg[o] for o in cfg_keys}

licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
    'mit': ('MIT License', 'OSI Approved :: MIT License'),
    'gpl2': ('GNU General Public License v2', 'OSI Approved :: GNU General Public License v2 (GPLv2)'),
    'gpl3': ('GNU General Public License v3', 'OSI Approved :: GNU General Public License v3 (GPLv3)'),
    'bsd3': ('BSD License', 'OSI Approved :: BSD License'),
}
statuses = [
    '1 - Planning',
    '2 - Pre-Alpha',
    '3 - Alpha',
    '4 - Beta',
    '5 - Production/Stable',
    '6 - Mature',
    '7 - Inactive'
]
py_versions = '3.9 3.10 3.11 3.12'.split()

requirements = cfg.get('requirements','').split()
if cfg.get('pip_requirements'): requirements += cfg.get('pip_requirements','').split()
min_python = cfg['min_python']
lic = licenses.get(cfg['license'].lower(), (cfg['license'], None))
dask_requirements = cfg['dask_requirements'].split()
ray_requirements = [
    req + " ; python_version < '3.12'" for req in cfg['ray_requirements'].split()
]
spark_requirements = cfg['spark_requirements'].split()
plotly_requirements = cfg['plotly_requirements'].split()
polars_requirements = cfg['polars_requirements'].split()
dev_requirements = cfg['dev_requirements'].split()
all_requirements = [
    *dask_requirements,
    *spark_requirements,
    *plotly_requirements,
    *polars_requirements,
    *dev_requirements,
    *ray_requirements,
]

ext_modules = [
    Pybind11Extension(
        name="statsforecast._lib",
        sources=glob.glob("src/*.cpp"),
        include_dirs=["include/statsforecast", "external_libs/eigen"],
        cxx_std=20,
    )
]
ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL", needs_recompile=naive_recompile).install()

setuptools.setup(
    name = 'statsforecast',
    license = lic[0],
    classifiers = [
        'Development Status :: ' + statuses[int(cfg['status'])],
        'Intended Audience :: ' + cfg['audience'].title(),
        'Natural Language :: ' + cfg['language'].title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]] + (['License :: ' + lic[1] ] if lic[1] else []),
    url = cfg['git_url'],
    package_dir={"": "python"},
    packages = setuptools.find_packages(where="python"),
    include_package_data = True,
    install_requires = requirements,
    extras_require={
        'dev': dev_requirements,
        'dask': dask_requirements,
        'ray': ray_requirements,
        'spark': spark_requirements,
        'plotly': plotly_requirements,
        'polars': polars_requirements,
        'all': all_requirements,
    },
    dependency_links = cfg.get('dep_links','').split(),
    python_requires  = '>=' + cfg['min_python'],
    long_description = open('README.md', encoding='utf8').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    entry_points = {
        'console_scripts': cfg.get('console_scripts','').split(),
        'nbdev': ['statsforecast=statsforecast._modidx:d']
    },
    ext_modules=ext_modules,
    **setup_cfg
)
