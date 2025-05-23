import torch
from torch import Tensor
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassConfusionMatrix
from typing import Optional
import matplotlib.pyplot as plt


def evaluate_classification(y_pred: Tensor,
                            y_true: Tensor,
                            num_classes: int,
                            average: str = 'macro',
                            cm_plot: bool = True,
                            class_names: Optional[list[str]] = None) -> dict:
    """
    Evaluate classification model using torchmetrics

    Sources:

    https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html
    https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
    https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html


    Args:
        y_pred (Tensor): Raw model outputs
        y_true (Tensor): Ground truth labels
        num_classes (int): Number of classes
        average (str): Averaging method for F1 score ('macro', 'weighted', etc)
        cm_plot (bool): Whether to show the plot of the confusion matrix

    Returns:
        dict: accuracy, f1, confusion_matrix

    """

    # Convert logits to predicted labels
    y_pred = torch.argmax(y_pred, dim=1)

    # Initialize Metrics
    acc = Accuracy(task="multiclass", num_classes=num_classes)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average=average)
    confmat = MulticlassConfusionMatrix(num_classes=num_classes)

    # Compute metrics
    acc_val = acc(y_pred, y_true)
    f1_val = f1(y_pred, y_true)
    cm = confmat(y_pred, y_true)

    if cm_plot:
        fig_, ax_ = confmat.plot(labels=class_names if class_names else None)
        fig_.show()
        plt.show()

    return {
        'Accuracy': acc_val.item(),
        'F1-score': f1_val.item(),
        'Confusion Matrix': cm
    }
