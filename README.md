# reCAPTCHA Solver 🛠️

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
    - [Installing Dependencies](#installing-dependencies)
    - [Testing](#testing)
- [Repo Structure](#repository-structure)
- [Usage Guide](#-usage-guide)
- [API Docs](#api-documentation)
    - [API Endpoints](#api-endpoints)
    - [Example API call](#example-api-call-and-response-format)
    - [Error Responses](#possible-error-responses)
- [Docker](#getting-started-with-docker)

---

This project is part of the **[Applied Machine Learning](https://ocasys.rug.nl/current/catalog/course/WBAI065-05#WBAI065-05.2024-2025.1)** course at the **University of Groningen**, developed by **Group 23**. Our project goal is to build an AI system that is able to automatically solve reCAPTCHA tests through **image classification techniques**.

---

### **Team Members:**

- **Katya Toncheva** (S5460786)  
- **Iva Ivanova** (S5614260)  
- **Beatrice Ursan-Racz** (S5572509)  
- **Sinan-Deniz Ceviker** (S5559308)  

Before getting started with our project, we encourage you to carefully read the sections below.

## Prerequisites
Make sure you have the following installed:

- **Pipenv**: Pipenv is used for dependency management. This tool enables users to easily create and manage virtual environments. To install Pipenv, use the following command:
    ```bash
    $ pip install --user pipenv
    ```
    For detailed installation instructions, [click here](https://pipenv.pypa.io/en/latest/installation.html).

## Getting Started
## Installing Dependencies
To install the project dependencies run:

```bash
pipenv install
```

This will automatically create a virtual environment.

To **activate** the virtual environment, run:

```bash
pipenv shell
```

To **deactivate** the virtual environment, run:

```bash
exit
```

## Testing
You can run all the unit and integration tests which use the standard _unittest_ Python module with the following command:

```bash
python -m unittest discover tests
```
If you wish to see additional details, run it in verbose mode:

```bash
python -m unittest discover -v tests
```

## Repository Structure

To make navigating through the repository easier, you can find its structure below, with additional comments.


```bash
├───data  # Stores the .csv dataset
├───models  # Stores the .pkl models
├───notebooks  # Empty
├───src
│   ├───data  # Data processing
│   ├───features # Evaluation class
│   └───models  # Model classes
├───reports
├───tests
│   ├───data  # Unit tests for data processing
│   ├───features  # Unit tests for evaluation
│   ├───integration  # Integration tests
│   └───models  # Unit tests for models
├───.gitignore
├───.pre-commit-config.yaml
├───main.py 
├───train_model.py
├───Pipfile  # Dependencies
├───Pipfile.lock
├───README.md  # Instructions
```

## **🚀 Usage Guide**

### Running the App

1. Activate pipenv environment (if not already activated)

```bash
pipenv shell
```

2. To launch any component of our project, run:
```bash
python main.py [OPTION]
```

Available list of options:
- --streamlit - Launches Streamlit UI
- --api - starts the FastAPI backend
- --train-simple-cnn - Trains the simple baseline model
- --train-main-cnn - Trains our main model

> If no argument has been passed, an interactive menu will appear to let you choose the action.
>
> Note: In order to use our trained model for predictions, please download it from the **[Releases](https://github.com/sinan2000/recaptcha/releases)** and place it in > the *models/* folder. This location can be changed by modifying the *MODELS_FOLDER* constant in constants.py


## API Documentation

### API Endpoints
- POST /predict: returns the predicted class id, name and probability of the class.

### Example API call and response format

You can make a call to the api using curl, by running the command below. Make sure to include a valid file path. The path can either be absolute (full) or relative to your
current location from command terminal.

```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@<path_to_file>"
```
You will get a response in the following format:

```json
 {
 "class_id":1,
 "class_name":"Bridge",
 "confidence": "99.9%"
 }
```

The API is stateless, initializing the model on launching, and caches responses for 1 hour.


After running the server, you can access the Documentation:

Interactive API docs (Swagger UI): http://localhost:8000/docs

ReDoc documentation: http://localhost:8000/redoc

These interfaces allow you to test predictions and inspect the request/response formats.

### Possible error Responses

| Status code  |    Description
|--------------|----------------------------------------------------------------|
|    200       |  Succesful Prediction                                          |
|    422       |  Validation error (eg. file not provided or malformed request) |
|    500       |  Internal server error                                         |
|    503       |  Model was not loaded - ensure you either trained or downloaded|

## Getting Started with Docker

1. Make sure [Docker](https://docs.docker.com/get-docker/) is installed on your machine and running.

2. Build the docker image by running the following command:
```bash
docker build -t recaptcha-app .
```

It may take a minute or two, depending on your internet speed.

3. Now, you just have to run it by opening 2 ports, for API and Streamlit access, by simply running this command:

```bash
docker run -it --shm-size=2g -p 8501:8501 -p 8000:8000 recaptcha-app
```

> Note: The docker image runs on linux, not having cuda support. Therefore only CPU torch will be available.