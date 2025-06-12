import torch
import os
from ..models.main_model.resnet_inspired_model_block import MainCNN

from src.constants import (
    MODELS_FOLDER,
    MAIN_MODEL_FILE_NAME,
    SIMPLE_MODEL_FILE_NAME
)


def load_simple_model(
     device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """
    Load the simple CNN model for image classification.

    Args:
        device (torch.device): The device to load the model on.

    Returns:
        torch.nn.Module: The loaded simple CNN model.
    """
    from ..models.simple_classifier_model import SimpleCNN
    model = SimpleCNN()
    path = get_model_path("simple")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_main_model(
     device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """
    Load the main CNN model for image classification.

    Args:
        device (torch.device): The device to load the model on.

    Returns:
        torch.nn.Module: The loaded main CNN model.
    """
    path = get_model_path("main")
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    model = MainCNN(
        n_layers=config['n_layers'],
        kernel_size=config['kernel_size'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def get_model_path(model_type: str) -> str | None:
    """
    Get the path to the model file based on the model type.

    Args:
        model_type (str): The type of model to load.

    Returns:
        str: The path to the model file.
    """
    if model_type == "simple":
        return os.path.join(MODELS_FOLDER, SIMPLE_MODEL_FILE_NAME)
    elif model_type == "main":
        return os.path.join(MODELS_FOLDER, MAIN_MODEL_FILE_NAME)
    return None
