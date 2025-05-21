# reCAPTCHA Solver with Multi-Task Learning 🛠️

This project is part of the **[Applied Machine Learning](https://ocasys.rug.nl/current/catalog/course/WBAI065-05#WBAI065-05.2024-2025.1)** course at the **University of Groningen**, developed by **Group 23**. Our project goal is to build an AI system that is able to automatically solve reCAPTCHA by combining **image classification** and **object detection** using **Multi-Task Learning**.

---

### **Team Members:**

- **Katya Toncheva** (S5460786)  
- **Iva Ivanova** (S5614260)  
- **Beatrice Ursan-Racz** (S5572509)  
- **Sinan-Deniz Ceviker** (S5559308)  

Before getting started with our project, we encourage you to carefully read the sections below.



## Prerequisites
Make sure you have the following installed:

- **Pipenv**: Pipenv is used for dependency management. This tools enables users to easily create and manage virtual environments. To install Pipenv, use the following command:
    ```bash
    $ pip install --user pipenv
    ```
    For detailed installation instructions, [click here](https://pipenv.pypa.io/en/latest/installation.html).

## Getting Started
### Setting up your own repository
1. Fork this repository.
2. Clone your fork locally.
3. Configure a remote pointing to the upstream repository to sync changes between your fork and the original repository.
   ```bash
   git remote add upstream https://github.com/ivopascal/Applied-ML-Template
   ```
   **Don't skip this step.** We might update the original repository, so you should be able to easily pull our changes.
   
   To update your forked repo follow these steps:
   1. `git fetch upstream`
   2. `git rebase upstream/main`
   3. `git push origin main`
      
      Sometimes you may need to use `git push --force origin main`. Only use this flag the first time you push after you rebased, and be careful as you might overwrite your teammates' changes.

### Pipenv
This tool is incredibly easy to use. Let's **install** our first package, which you will all need in your projects.

```bash
pipenv install <package-name>
```

After running this command, you will notice that two files were modified, namely, _Pipfile_ and _Pipfile.lock_. _Pipfile_ is the configuration file that specifies all the dependencies in your virtual environment.

To **uninstall** a package, you can run the command:
```bash
pipenv uninstall <package-name>
```

To **activate** the virtual environment, run `pipenv shell`. You can now use the environment as you wish. To **deactivate** the environment run the command `exit`.

If you **already have access to a Pipfile**, you can install the dependencies using `pipenv install`.

For a comprehensive list of commands, consult the [official documentation](https://pipenv.pypa.io/en/latest/cli.html).

### Unit testing
You are expected to test your code using unit testing, which is a technique where small individual components of your code are tested in isolation.

An **example** is given in _tests/test_main.py_, which uses the standard _unittest_ Python module to test whether the function _hello_world_ from _main.py_ works as expected.

To run all the tests developed using _unittest_, simply use:
```bash
python -m unittest discover tests
```
If you wish to see additional details, run it in verbose mode:
```bash
python -m unittest discover -v tests
```

## Get Coding
You are now ready to start working on your projects.

We recommend following the same folder structure as in the original repository. This will make it easier for you to have cleaner and consistent code, and easier for us to follow your progress and help you.

Your repository should look something like this:
```bash
├───data  # Stores .csv
├───models  # Stores .pkl
├───notebooks  # Contains experimental .ipynbs
├───project_name
│   ├───data  # For data processing, not storing .csv
│   ├───features
│   └───models  # For model creation, not storing .pkl
├───reports
├───tests
│   ├───data
│   ├───features
│   └───models
├───.gitignore
├───.pre-commit-config.yaml
├───main.py
├───train_model.py
├───Pipfile
├───Pipfile.lock
├───README.md
```

**Good luck and happy coding! 🚀**