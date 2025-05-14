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

    if eval_classification or eval_detection:
        for batch in tqdm(test_loader, desc="Evaluating"):
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            elif isinstance(batch, dict):
                images = batch["images"]
                labels = batch["labels"]
            else:
                raise ValueError("Unsupported batch format")

            images = images.to(device)

            with torch.no_grad():
                output = model(images)

            if eval_classification:
                if isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output[0]
                all_preds.append(logits)
                all_targets.append(labels.to(device))

            if eval_detection:
                if isinstance(output, torch.Tensor):
                    pred_boxes = output
                else:
                    pred_boxes = output[1]

                for i in range(len(images)):
                    results.setdefault("detection_preds", []).append({
                        "boxes": pred_boxes[i].unsqueeze(0),
                        "scores": torch.tensor([1.0]),  # placeholder
                        "labels": torch.tensor([0])    # placeholder
                    })
                    results.setdefault("detection_targets",
                                       []).append(labels[i])

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

    if eval_detection:
        map_results = dm.evaluate_map(
            results["detection_preds"],
            results["detection_targets"]
        )
        results.update(map_results)
        del results["detection_preds"]
        del results["detection_targets"]

    print("\n--- Evaluation Results ---")
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            val = val.item()
        print(f"{key}: {val:.4f}")

    return results
