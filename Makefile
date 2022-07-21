.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard nbs/*.ipynb)

all: statsforecast docs

statsforecast: $(SRC)
	nbdev_build_lib
	touch statsforecast

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs --mk_readme False
	nbdev_build_docs --fname "examples/*.ipynb" --mk_readme False
	touch docs

build_docs:
	nbdev_build_docs --mk_readme False
	nbdev_build_docs --fname "examples/*.ipynb" --mk_readme False

test:
	nbdev_test_nbs

release: pypi conda_release
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

nbdev_flow:
	nbdev_install_git_hooks && nbdev_build_lib \
				&& nbdev_build_docs \
				&& nbdev_clean_nbs --clear_all True --fname "nbs/*" \
				&& nbdev_diff_nbs \
				&& nbdev_test_nbs --timing
