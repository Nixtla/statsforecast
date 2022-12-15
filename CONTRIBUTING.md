# How to contribute

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

## Do you have a feature request?

* Ensure that it hasn't been yet implemented in the `main` branch of the repository and that there's not an Issue requesting it yet.
* Open a new issue and make sure to describe it clearly, mention how it improves the project and why its useful.

## Do you want to fix a bug or implement a feature?

Bug fixes and features are added through pull requests (PRs).

##  PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Ensure that your PR includes a test that fails without your patch, and passes with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

### Local setup for working on a PR

#### 1. Clone the repository
* HTTPS: `git clone https://github.com/Nixtla/statsforecast.git`
* SSH: `git clone git@github.com:Nixtla/statsforecast.git`
* GitHub CLI: `gh repo clone Nixtla/statsforecast`

#### 2. Set up a conda environment
The repo comes with an `environment.yml` file which contains the libraries needed to run all the tests. In order to set up the environment you must have `conda` installed, we recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Once you have `conda` go to the top level directory of the repository and run:
```
conda env create -f environment.yml
```

#### 3. Install the library
Once you have your environment setup, activate it using `conda activate statsforecast` and then install the library in editable mode using `pip install -e ".[dev]"`

#### 4. Install git hooks
Before doing any changes to the code, please install the git hooks that run automatic scripts during each commit and merge to strip the notebooks of superfluous metadata (and avoid merge conflicts).
```
nbdev_install_hooks
```

### Preview Changes
You can preview changes in your local browser before pushing by using the `nbdev_preview`. This command also installs Quarto if not available. 

### Building the library
The library is built using the notebooks contained in the `nbs` folder. If you want to make any changes to the library you have to find the relevant notebook, make your changes and then call `nbdev_export`.

### Running tests
If you're working on the local interface you can just use `nbdev_test`. 

### Linters
This project uses a couple of linters to validate different aspects of the code. Before opening a PR, please make sure that it passes all the linting tasks by following the next steps.

#### Run the linting tasks
* `mypy statsforecast/`
* `flake8 --select=F statsforecast/`

### Cleaning notebooks
Since the notebooks output cells can vary from run to run (even if they produce the same outputs) the notebooks are cleaned before committing them. Please make sure to run `nbdev_clean` before committing your changes.

## Do you want to contribute to the documentation?

* Docs are automatically created from the notebooks in the `nbs` folder.
* In order to modify the documentation:
    1. Find the relevant notebook.
    2. Make your changes.
    3. Run all cells.
    4. If you are modifying library notebooks (not in `nbs/examples`), clean all outputs using `Edit > Clear All Outputs`.
    5. Run `nbdev_preview`.
    6. Clean the notebook metadata using `nbdev_clean`.
