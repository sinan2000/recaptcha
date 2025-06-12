import os
from pathlib import Path


IMAGE_CHANNELS = 3
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)


BASE_DIR = Path(__file__).resolve().parent.parent
# Folder where models are saved
MODELS_FOLDER = os.path.join(BASE_DIR, "models", "final")
MAIN_MODEL_FILE_NAME = "main_model.pt"
SIMPLE_MODEL_FILE_NAME = "simple_model.pt"
OPTIMIZER_FILE_NAME = "optimizer.pt"
SCHEDULER_FILE_NAME = "scheduler.pt"
