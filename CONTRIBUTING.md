# Step-by-step Contribution Guide

> This document contains instructions for collaborating on the different libraries of Nixtla.

Sometimes, diving into a new technology can be challenging and overwhelming. We've been there too, and we're more than ready to assist you with any issues you may encounter while following these steps. Don't hesitate to reach out to us on [Slack](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ). Just give fede a ping, and she'll be glad to assist you.

## Table of Contents 📚

1. [Prerequisites](#prerequisites)
2. [Git `fork-and-pull` worklow](#git-fork-and-pull-worklow)
3. [Set Up a Conda Environment](#set-up-a-conda-environment)
4. [Install required libraries for development](#install-required-libraries-for-development)
5. [Start editable mode](#start-editable-mode)
6. [Set Up your Notebook based development environment](#set-up-your-notebook-based-development-environment)
7. [Start Coding](#start-coding)
8. [Example with Screen-shots](#example-with-screen-shots)

## Prerequisites 

- *GitHub*: You should already have a GitHub account and a basic understanding of its functionalities. Alternatively check [this guide](https://docs.github.com/en/get-started).
- *Python*: Python should be installed on your system. Alternatively check [this guide](https://www.python.org/downloads/). 
- *conda*: You need to have conda installed, along with a good grasp of fundamental operations such as creating environments, and activating and deactivating them.  Alternatively check [this guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

## Git `fork-and-pull` worklow

**1. Fork the Project:** 
Start by forking the Nixtla repository to your own GitHub account. This creates a personal copy of the project where you can make changes without affecting the main repository.

**2. Clone the Forked Repository**
Clone the forked repository to your local machine using `git clone https://github.com/<your-username>/nixtla.git`. This allows you to work with the code directly on your system. 

**3. Create a Branch:** 

Branching in GitHub is a key strategy for effectively managing and isolating changes to your project. It allows you to segregate work on different features, fixes, and issues without interfering with the main, production-ready codebase.

1. *Main Branch*: The default branch with production-ready code.

2. *Feature Branches*: For new features, create branches prefixed with 'feature/', like `git checkout -b feature/new-model`.

3. *Fix Branches*: For bug fixes, use 'fix/' prefix, like `git checkout -b fix/forecasting-bug`.

4. *Issue Branches*: For specific issues, use `git checkout -b issue/issue-number` or `git checkout -b issue/issue-description`.

After testing, branches are merged back into the main branch via a pull request, and then typically deleted to maintain a clean repository. You can read more about github and branching [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).

##  Set Up a Conda Environment

> If you want to use Docker or Codespaces, let us know opening an issue and we will set you up.

Next, you'll need to set up a [Conda](https://docs.conda.io/en/latest/) environment. Conda is an open-source package management and environment management system that runs on Windows, macOS, and Linux. It allows you to create separate environments containing files, packages, and dependencies that will not interact with each other.

First, ensure you have Anaconda or Miniconda installed on your system. Alternatively checkout these guides: [Anaconda](https://www.anaconda.com/), [Miniconda](https://docs.conda.io/en/latest/miniconda.html), and [Mamba](https://mamba.readthedocs.io/en/latest/).

Then, you can create a new environment using `conda create -n nixtla-env python=3.10`. 

You can also use mamba for creating the environment (mamba is faster than Conda) using `mamba create -n nixtla-env python=3.10`. 

Activate your new environment with `conda activate nixtla-env`. 

## Install required libraries for development

The `environment.yml` file contains all the dependencies required for the project. To install these dependencies, use the `mamba` package manager, which offers faster package installation and environment resolution than Conda. If you haven't installed `mamba` yet, you can do so using `conda install mamba -c conda-forge`. Run the following command to install the dependencies:

```
mamba env update -f environment.yml
```

## Start editable mode

Install the library in editable mode using `pip install -e ".[dev]"`. 

This means the package is linked directly to the source code, allowing any changes made to the source code to be immediately reflected in your Python environment without the need to reinstall the package. This is useful for testing changes during package development.

## Set Up your Notebook based development environment

Notebook-based development refers to using interactive notebooks, such as Jupyter Notebooks, for coding, data analysis, and visualization. Here's a brief description of its characteristics:

1. **Interactivity**: Code in notebooks is written in cells which can be run independently. This allows for iterative development and testing of small code snippets.

2. **Visualization**: Notebooks can render charts, tables, images, and other graphical outputs within the same interface, making it great for data exploration and analysis.

3. **Documentation**: Notebooks support Markdown and HTML, allowing for detailed inline documentation. Code, outputs, and documentation are in one place, which is ideal for tutorials, reports, or sharing work.

For notebook based development you'll need `nbdev` and a notebook editor (such as VS Code, Jupyter Notebook or Jupyter Lab). `nbdev` and jupyter have been installed in the previous step. If you use VS Code follow [this tutorial](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

[nbdev](https://github.com/fastai/nbdev) makes debugging and refactoring your code much easier than in traditional programming environments since you always have live objects at your fingertips. `nbdev` also promotes software engineering best practices because tests and documentation are first class.

All your changes must be written in the notebooks contained in the library (under the `nbs` directory). Once a specific notebook is open (more details to come), you can write your Python code in cells within the notebook, as you would do in a traditional Python development workflow. You can break down complex problems into smaller parts, visualizing data, and documenting your thought process. Along with your code, you can include markdown cells to add documentation directly in the notebook. This includes explanations of your logic, usage examples, and more. Also, `nbdev` allows you to write [tests inline](https://nbdev.fast.ai/tutorials/best_practices.html#document-error-cases-as-tests) with your code in your notebook. After writing a function, you can immediately write tests for it in the following cells.

Once your code is ready, `nbdev` can automatically convert your notebook into Python scripts. Code cells are converted into Python code, and markdown cells into comments and docstrings.

## Start Coding

Open a jupyter notebook using `jupyter lab` (or VS Code).

1. **Make Your Changes:** Make changes to the codebase, ensuring your changes are self-contained and cohesive.

2. **Commit Your Changes:** Add the changed files using `git add [your_modified_file_0.ipynb] [your_modified_file_1.ipynb]`, then commit these changes using `git commit -m "Your descriptive commit message"`.

3. **Push Your Changes:** 
Push your changes to the remote repository on GitHub with `git push origin feature/your-feature-name`.

4. **Open a Pull Request:** Open a pull request from your new branch on the Nixtla repository on GitHub. Provide a thorough description of your changes when creating the pull request.

5. **Wait for Review:** The maintainers of the Nixtla project will review your changes. Be ready to iterate on your contributions based on their feedback.

Remember, contributing to open-source projects is a collaborative effort. Respect the work of others, welcome feedback, and always strive to improve. Happy coding!

> Nixtla offers the possibility of assisting with stipends for computing infrastructure for our contributors. If you are interested, please join our [slack](https://nixtlacommunity.slack.com/join/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ#/shared-invite/email) and write to fede or Max.

You can find a detailed step by step buide with screen-shots below.

## Example with Screen-shots

### 1. Create a fork of the mlforecast repo
The first thing you need to do is create a fork of the GitHub repository to your own account:
![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/af767f5b-66f1-4068-9dd2-917096285ae9)

Your fork on your account will look like this:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/912b848d-d3d1-4f79-9d5b-10dd66e1bd35)

In that repository, you can make your changes and then request to have them added to the main repo.

### 2. Clone the repository

In this tutorial, we are using Mac (also compatible with other Linux distributions). If you are a collaborator of Nixtla, you can request an AWS instance to collaborate from there. If this is the case, please reach out to Max or Fede on [Slack](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ) to receive the appropriate access. We also use Visual Studio Code, which you can download from [here](https://code.visualstudio.com/download).

Once the repository is created, you need to clone it to your own computer. Simply copy the repository URL from GitHub as shown below:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/1331354e-ac24-4222-82f1-71df7077f2e0)

Then open Visual Studio Code, click on "Clone Git Repository," and paste the line you just copied into the top part of the window, as shown below:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/982827d5-89a6-43d4-8bb8-85bd1758bc10)

Select the folder where you want to copy the repository:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/c1a169e6-df27-41fb-84ee-a441a149e3d6)

And choose to open the cloned repository:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/00140c15-237e-4afa-a47d-078a1afbbac0)

You will end up with something like this:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/ea4aed6f-2000-4ec8-a242-36b9dfd68d26)

### 3. Create the Conda environment

Open a terminal within Visual Studio Code, as shown in the image:

<img width="1423" alt="image" src="https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/9b3ed42f-1a68-450c-bffd-a7cee40bb781">

You can use conda but we highly recommend using Mamba to speed up the creation of the Conda environment. To install it, simply use `conda install mamba -c conda-forge` in the terminal you just opened:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/08482b00-9434-46f0-9452-c3f4920eca6d)

Create an empty environment named `mlforecast` with the following command: `mamba create -n mlforecast python=3.10`:
![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/5e9032e8-3f5b-4a1c-93e7-3d390d5f73f1)

Activate the newly created environment using `conda activate mlforecast`:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/803ae2b7-8369-4a24-9b7c-9326d52c13ef)

Install the libraries within the environment file `environment.yml` using `mamba env update -f environment.yml`:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/e9672d58-b477-4963-9751-277c944a4d8a)

Now install the library to make interactive changes and other additional dependencies using `pip install -e ".[dev]"`:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/501c8223-862d-40a9-8f2d-ecdaceaeaedb)

### 4. Make the changes you want.

In this section, we assume that we want to increase the default number of windows used to create prediction intervals from 2 to 3. The first thing we need to do is create a specific branch for that change using `git checkout -b [new_branch]` like this:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/7884f89a-ecc6-4200-8176-6a9b9f7c0aa2)

Once created, open the notebook you want to modify. In this case, it's `nbs/utils.ipynb`, which contains the metadata for the prediction intervals. After opening it, click on the environment you want to use (top right) and select the `mlforecast` environment:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/0a0a8285-9344-471e-b699-8bc13159e3a8)

Next, execute the notebook and make the necessary changes. In this case, we want to modify the `PredictionIntervals` class:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/0a614d14-52c9-4ba1-88e5-02e19661cae7)

We will change the default value of `n_window` from 2 to 3:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/af31a79d-0165-4c79-94bc-f411ec67b3ea)

Once you have made the change and performed any necessary validations, it's time to convert the notebook to Python modules. To do this, simply use `nbdev_export` in the terminal.

You will see that the `mlforecast/utils.py` file has been modified (the changes from `nbs/utils.ipynb` are reflected in that module). Before committing the changes, we need to clean the notebooks using the command `./action_files/clean_nbs` and verify that the linters pass using `./action_files/lint`:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/cecf76a1-c025-4b6f-97c0-584394a3f56a)

Once you have done the above, simply add the changes using `git add nbs/utils.ipynb mlforecast/utils.py`:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/107689d8-a270-4621-ac5d-9d077d9203c3)

Create a descriptive commit message for the changes using `git commit -m "[description of changes]"`:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/034578e2-d63c-4d74-a96f-99f405288326)

Finally, push your changes using `git push`:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/49c6851c-949b-4ca7-ac38-6b17ec103437)


### 5. Create a pull request.

In GitHub, open your repository that contains your fork of the original repo. Once inside, you will see the changes you just pushed. Click on "Compare and pull request":

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/c3d2ce0d-5fc9-4356-87d3-51b32e72524a)

Include an appropriate title for your pull request and fill in the necessary information. Once you're done, click on "Create pull request".

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/7de9c461-ad49-4fc2-b648-507662466851)

Finally, you will see something like this:

![image](https://github.com/Nixtla/how-to-contribute-nixtlaverse/assets/10517170/846c0b97-46d2-492b-a58e-3e9f669c1632)


## Notes
- This file was generated using [this file](https://github.com/Nixtla/nixtla-commons/blob/main/docs/contribute/step-by-step.md). Please change that file if you want to enhance the document.
