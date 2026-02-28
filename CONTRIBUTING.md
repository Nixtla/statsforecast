# Step-by-step Contribution Guide

> This document contains instructions for collaborating on the different libraries of Nixtla.

Sometimes, diving into a new technology can be challenging and overwhelming. We've been there too, and we're more than ready to assist you with any issues you may encounter while following these steps. Don't hesitate to reach out to us on [Slack](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ).

## Table of Contents 📚

- [Step-by-step Contribution Guide](#step-by-step-contribution-guide)
  - [Table of Contents 📚](#table-of-contents-)
  - [Prerequisites](#prerequisites)
  - [Git `fork-and-pull` worklow](#git-fork-and-pull-worklow)
  - [Set Up a Virtual Environment](#set-up-a-virtual-environment)
  - [Install required libraries for development](#install-required-libraries-for-development)
    - [Setup pre-commit hooks](#setup-pre-commit-hooks)
  - [Start editable mode](#start-editable-mode)
    - [Re-compiling the shared library](#re-compiling-the-shared-library)
  - [Set Up your Notebook based development environment](#set-up-your-notebook-based-development-environment)
  - [Running tests](#running-tests)
  - [Viewing documentation locally](#viewing-documentation-locally)
    - [Install `quarto`](#install-quarto)
    - [Install mintlify](#install-mintlify)

## Prerequisites

- **GitHub**: You should already have a GitHub account and a basic understanding of its functionalities. Alternatively check [this guide](https://docs.github.com/en/get-started).
- **uv**: You need to have `uv` installed. You can refer to the [docs](https://docs.astral.sh/uv/getting-started/installation/) in order to install it.

## Git `fork-and-pull` worklow

**1. Fork the Project:**
Start by forking the Nixtla repository to your own GitHub account. This creates a personal copy of the project where you can make changes without affecting the main repository.

**2. Clone the Forked Repository**
Clone the forked repository to your local machine using `git clone --recursive https://github.com/<your-username>/statsforecast.git`. This allows you to work with the code directly on your system.

**3. Create a Branch:**

Branching in GitHub is a key strategy for effectively managing and isolating changes to your project. It allows you to segregate work on different features, fixes, and issues without interfering with the main, production-ready codebase.

1. *Main Branch*: The default branch with production-ready code.

2. *Feature Branches*: For new features, create branches prefixed with 'feature/', like `git checkout -b feature/new-model`.

3. *Fix Branches*: For bug fixes, use 'fix/' prefix, like `git checkout -b fix/forecasting-bug`.

4. *Issue Branches*: For specific issues, use `git checkout -b issue/issue-number` or `git checkout -b issue/issue-description`.

After testing, branches are merged back into the main branch via a pull request, and then typically deleted to maintain a clean repository. You can read more about github and branching [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).

## Set Up a Virtual Environment

> If you want to use Docker or Codespaces, let us know opening an issue and we will set you up.

Next, you'll need to create a virtual environment. uv is an open-source package management and environment management system that runs on Windows, macOS, and Linux. It allows you to create isolated environments with the dependencies required for a specific project.

First, ensure you have uv installed on your system. Alternatively, refer to the [installation docs](https://docs.astral.sh/uv/getting-started/installation/).

Then, you can create a new environment using `uv venv`. This will use your default python interpreter, you can also define a specific python version (which will be downloaded if you don't have it already) by providing the `--python` flag. For example, `uv venv --python 3.12`.

Activate your new environment with `source .venv/bin/activate` for MacOS and Linux or `.\.venv\Scripts\activate` for Windows.

## Install required libraries for development

The `setup.py` file contains all the dependencies required for the project. To install these dependencies you can use `uv pip install -r setup.py --extra dev`

### Setup pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to ease the development process, which run some checks before you make a commit to have a faster feedback loop.

To setup the pre-commit hooks run: `pre-commit install`

## Start editable mode

Install the library in editable mode using `uv pip install --no-build-isolation -e .` (this requires a C++ compiler).

> [!NOTE]
> When using `--no-build-isolation`, build dependencies are not installed automatically. If you encounter `error: invalid command 'bdist_wheel'`, install `wheel` first: `uv pip install wheel`

By using the `-e` flag the package is linked directly to the source code, allowing any changes made to the source code to be immediately reflected in your Python environment without the need to reinstall the package. This is useful for testing changes during package development.

### Re-compiling the shared library

If you're working on the C++ code, you'll need to re-compile the shared library, which can be done with: `python setup.py build_ext --inplace` (this will compile it into the `build` directory and copy it to the python package location).

## Set Up your Notebook based development environment

Notebooks are only used in the project for how-to-guides and code-walkthroughs.

## Running tests

To run the tests, run

```sh
uv run pytest
```

## Viewing documentation locally

The new documentation pipeline relies on `quarto`, `mintlify` and `griffe2md`.

### Install `quarto`

Install `quarto` from &rarr; [this link](https://quarto.org/docs/get-started/)

### Install mintlify

> [!NOTE]
> Please install Node.js before proceeding.

```sh
npm i -g mint
```

For additional instructions, you can read about it &rarr; [this link](https://mintlify.com/docs/installation).

```sh
uv pip install -e '.[dev, docs]'
make all_docs
```

Finally to view the documentation

```sh
make preview_docs
```

- The docs are automatically generated from the docstrings in the `python/statsforecast` folder.
- To contribute, ensure your docstrings follow the Google style format.
- Once your docstring is correctly written, the documentation framework will scrape it and regenerate the corresponding `.mdx` files and your changes will then appear in the updated docs.
- Make an appropriate entry in the `docs/mintlify/docs.json` file.
- Run `make all_docs` to regenerate the documentation.
- Run `make preview_docs` to view and test the documentation locally.
