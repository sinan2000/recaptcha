import numpy as np
import torch
import torch.nn.functional as F


class WrappedModel(torch.nn.Module):
    """ Wrapper class to make model compatible with captum """
    def __init__(self, model: torch.nn.Module) -> None:
        """ Constructor for WrappedModel class.

        Args:
            model (torch.nn.Module): Main CNN model for image classification.
        """
        super().__init__()
        self.model = model
        self.eval()
        self.model.eval()

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(
                next(self.model.parameters()).device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()
