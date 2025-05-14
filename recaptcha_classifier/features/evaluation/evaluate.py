import recaptcha_classifier.features.evaluation.classification_metrics as cm
import recaptcha_classifier.features.evaluation.detection_metrics as dm
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   device: torch.device,
                   num_classes: int = None,
                   class_names: list[str] = None,
                   eval_classification: bool = True,
                   eval_detection: bool = False) -> dict:
    """
    Unified evaluation function for classification and object detection models.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to evaluate on.
        num_classes (int): Required for classification metrics.
        class_names (list[str], optional): Optional class names for confusion
        matrix plot.
        eval_classification (bool): Whether to compute classification metrics.
        eval_detection (bool): Whether to compute object detection metrics.

    Returns:
        dict: Dictionary containing selected evaluation results.
    """
    model.eval()
    model.to(device)

    results = {}
    all_preds = []
    all_targets = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        images, labels = batch
        images = images.to(device)
        output = model(images)

        logits = output if isinstance(output, torch.Tensor) else output[0]
        pred_boxes = output if isinstance(output, torch.Tensor) else output[1]

        # Classification part
        if eval_classification:
            # Extract classification labels if detection targets format
            if isinstance(labels, list):  # detection targets (list of dicts)
                y_labels = torch.tensor([t["labels"].item() for t in labels])
            else:
                y_labels = labels

            all_preds.append(logits)
            all_targets.append(y_labels.to(device))

        # Detection part
        if eval_detection:
            for i in range(len(images)):
                detection_pred = {
                    "boxes": pred_boxes[i].unsqueeze(0),
                    "scores": torch.tensor([1.0]),  # Dummy score
                    "labels": torch.tensor([0])     # Dummy label
                }
                results.setdefault("detection_preds",
                                   []).append(detection_pred)
                results.setdefault("detection_targets", []).append(labels[i])

    # Compute classification metrics
    if eval_classification:
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)

        class_results = cm.evaluate_classification(
            y_pred=y_pred,
            y_true=y_true,
            num_classes=num_classes,
            class_names=class_names,
            cm_plot=True
        )
        results.update(class_results)

    # Compute detection metrics
    if eval_detection:
        map_results = dm.evaluate_map(
            results["detection_preds"],
            results["detection_targets"]
        )
        results.update(map_results)
        del results["detection_preds"]
        del results["detection_targets"]

    # Print results safely
    print("\n--- Evaluation Results ---")
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            if val.ndim == 0:
                val = val.item()
            else:
                val = val.tolist()
        print(f"{key}: {val}")

    return results
