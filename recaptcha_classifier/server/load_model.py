import torch
import os
from ..models.main_model.model_class import MainCNN

from recaptcha_classifier.constants import (
    MODELS_FOLDER,
    MAIN_MODEL_FILE_NAME,
    SIMPLE_MODEL_FILE_NAME
)


def load_simple_model(device: torch.device = torch.device("cpu")):
    """
    Load the simple CNN model for image classification.
    """
    from ..models.simple_classifier_model import SimpleCNN
    model = SimpleCNN()
    path = os.path.join(MODELS_FOLDER, SIMPLE_MODEL_FILE_NAME)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
    
def load_main_model(device: torch.device = torch.device("cpu")):
    """
    Load the main CNN model for image classification.
    """
    path = os.path.join(MODELS_FOLDER, MAIN_MODEL_FILE_NAME)
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