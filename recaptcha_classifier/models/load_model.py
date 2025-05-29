import torch
from torchvision import transforms
from PIL import Image

from .main_model.model_class import MainCNN
from recaptcha_classifier.detection_labels import DetectionLabels

from recaptcha_classifier.constants import (
    MAIN_MODEL_FOLDER,
    MAIN_MODEL_FILE_NAME,
    SIMPLE_MODEL_FOLDER,
    SIMPLE_MODEL_FILE_NAME
)


def load_simple_model(device: torch.device = torch.device("cpu")):
    """
    Load the simple CNN model for image classification.
    """
    from .simple_classifier_model import SimpleCNN
    model = SimpleCNN()
    path = SIMPLE_MODEL_FOLDER + "/" + SIMPLE_MODEL_FILE_NAME
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
    
def load_main_model(device: torch.device = torch.device("cpu")):
    """
    Load the main CNN model for image classification.
    """
    path = MAIN_MODEL_FOLDER + "/" + MAIN_MODEL_FILE_NAME
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