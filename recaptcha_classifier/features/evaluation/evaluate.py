from .classification_metrics import evaluate_classification
import torch
from tqdm import tqdm
from recaptcha_classifier.detection_labels import DetectionLabels


@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   device: torch.device,
                   class_names: list[str] = DetectionLabels.all(),
                   plot_cm: bool = False) -> dict:
    """
    Evaluation function for classification models.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to evaluate on.
        class_names (list[str]): Class names for confusion matrix.
        plot_cm (bool): Whether to show a plot of the confusion matrix.

    Returns:
        dict: Dictionary containing evaluation results.
    """
    model.eval()
    model.to(device)

    results = {}
    all_preds = []
    all_targets = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        output = model(images)

        logits = output if isinstance(output, torch.Tensor) else output[0]

        all_preds.append(logits)
        all_targets.append(labels.to(device))

    # Compute classification metrics
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    class_results = evaluate_classification(
        y_pred=y_pred,
        y_true=y_true,
        class_names=class_names,
        cm_plot=plot_cm
    )
    results.update(class_results)

    print("\n--- Evaluation Results ---")
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            if val.ndim == 0:
                val = val.item()
            else:
                val = val.tolist()
        print(f"{key}: {val}")

    return results
