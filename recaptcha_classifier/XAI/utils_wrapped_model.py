import numpy as np
import torch
import torch.nn.functional as F


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.eval()
        self.model.eval()

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        # Convert to torch tensor if needed

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(next(self.model.parameters()).device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()