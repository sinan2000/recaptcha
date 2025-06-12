FROM python:3.12-slim

# system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# install pipenv
RUN pip install pipenv

# set the working directory
WORKDIR /app

# copy the Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock /app/

# install dependencies directly in python, not with venv
RUN pipenv install --deploy --ignore-pipfile --system

# copy the rest of the application code
COPY . /app/

# set the entrypoint
ENTRYPOINT ["python", "main.py"]