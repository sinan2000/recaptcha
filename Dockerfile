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

# install dependencies, ensuring torch is used from the cpu version (no CUDA)
RUN pipenv install --deploy --ignore-pipfile && \
    pipenv run pip uninstall -y torch torchvision torchaudio && \
    pipenv run pip install torch torchvision torchaudio --no-deps --index-url https://download.pytorch.org/whl/cpu

# copy the rest of the application code
COPY . /app/

# set the entrypoint
ENTRYPOINT ["pipenv", "run", "python", "main.py"]