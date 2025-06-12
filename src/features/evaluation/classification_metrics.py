import torch
import pandas as pd
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassAccuracy
)
from typing import Optional
from src.detection_labels import DetectionLabels
import matplotlib.pyplot as plt


def evaluate_classification(y_pred: Tensor,
                            y_true: Tensor,
                            # num_classes: int = len(DetectionLabels.all()),
                            device: torch.device = torch.device(
                                "cuda" if torch.cuda.is_available()
                                else "cpu"),
                            average: str = 'weighted',
                            cm_plot: bool = True,
                            class_names: Optional[list[str]] = None,
                            save_path: str = "predictions.csv"
                            ) -> dict:
    """
    Evaluate classification model using torchmetrics

    Sources:
    https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html
    https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
    https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html


    Args:
        y_pred (Tensor): Raw model outputs
        y_true (Tensor): Ground truth labels
        device (torch.device): Device to use
        num_classes (int): Number of classes
        average (str): Averaging method for F1 score ('macro', 'weighted', etc)
        cm_plot (bool): Whether to show the plot of the confusion matrix

    Returns:
        dict: accuracy, f1, confusion_matrix
    """
    num_classes = len(class_names)

    logits = y_pred.to(device)

    # Convert logits to predicted labels
    y_pred = torch.argmax(y_pred, dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    # Initialize Metrics
    acc = Accuracy(
        task="multiclass", num_classes=num_classes).to(device=device)
    f1 = F1Score(
        task="multiclass", num_classes=num_classes, average=average).to(
            device=device)
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(
        device=device)
    topk_acc = MulticlassAccuracy(
        top_k=3, num_classes=num_classes).to(device=device)

    probs = F.softmax(logits, dim=1)
    # Entropy per sample
    entropy_vals = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    entropy_mean = entropy_vals.mean().item()
    entropy_sem = entropy_vals.std(unbiased=True).item() / (
        len(entropy_vals) ** 0.5)
    entropy_ci = 1.96 * entropy_sem

    # Variance per sample
    variance_vals = probs.var(dim=1)
    variance_mean = variance_vals.mean().item()

    # Compute metrics
    acc_val = acc(y_pred, y_true)
    f1_val = f1(y_pred, y_true)
    topk_acc_val = topk_acc(logits, y_true)
    cm = confmat(y_pred, y_true)

    if cm_plot:
        fig_, ax_ = confmat.plot(labels=class_names if class_names else None)
        plt.show()
        
    if save_path:
        df = pd.DataFrame({
            "true": y_true.cpu().numpy(),
            "predicted": y_pred.cpu().numpy()
        })
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")

    return {
        'Accuracy': acc_val.item(),
        'F1-score': f1_val.item(),
        'Confusion Matrix': cm,
        'Top-3 Accuracy': topk_acc_val.item(),
        'Mean Entropy': entropy_mean,
        'Entropy SEM': entropy_sem,
        'Entropy 95% CI': (
            entropy_mean - entropy_ci, entropy_mean + entropy_ci),
        'Mean Softmax Variance': variance_mean,
    }
