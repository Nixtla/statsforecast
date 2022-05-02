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
	nbdev_build_docs
	touch docs

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
				&& nbdev_clean_nbs \
				&& nbdev_diff_nbs \
				&& nbdev_test_nbs --timing