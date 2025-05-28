import torch
from recaptcha_classifier import MainCNN, DetectionLabels

def load_model(path="models/model.pt"):
    model = MainCNN(
        n_layers=2,
        kernel_size=3,
        num_classes=len(DetectionLabels),
    )
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model