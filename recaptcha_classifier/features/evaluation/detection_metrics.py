import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def yolo_to_corners(x_center: float,
                    y_center: float,
                    width: float,
                    height: float,
                    img_w: int,
                    img_h: int) -> list[float]:
    """
    Converts a bounding box from (normalized) YOLO format to coordinates

    Source:
    https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h

    YOLO format:
        - x_center, y_center: center of the box (normalized between 0 and 1)
        - width, height: size of the box (normalized between 0 and 1)

    This function scales the coordinates to the actual image size and returns:
        - [x1, y1, x2, y2] = top-left and bottom-right corners of the box
        (in pixels)

    Args:
        x_center (float): Normalized x-coordinate of the box center.
        y_center (float): Normalized y-coordinate of the box center.
        width (float): Normalized width of the box.
        height (float): Normalized height of the box.
        img_w (int): Width of the image in pixels.
        img_h (int): Height of the image in pixels.

    Returns:
        List[float]: Box in corner format [x1, y1, x2, y2] in pixel
        coordinates.
    """

    # Scale to pixel coordinates (from normalized state)
    x_center = x_center * img_w
    y_center = y_center * img_h
    width = width * img_w
    height = height * img_h

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2]


def compute_iou(box1: list[float],
                box2: list[float]) -> float:
    """
    Computes the Intersection over Union (IoU) value, to evaluate
    bounding box accuracy for object detection. Both box lists are in
    the following coordinate format [x1, y1, x2, y2].

    Source: https://www.v7labs.com/blog/intersection-over-union-guide

    Args:
        box1 (list[float]): Predicted box coordinates
        box2 (list[float]): Ground Truth box coordinates


    Returns:
        float: IoU value between 0 and 1, where 1 means perfect overlap
        and 0 means no overlap.

    """
    # Left side of the overlap
    xA = max(box1[0], box2[0])
    # Top side of the overlap
    yA = max(box1[1], box2[1])
    # Right side of the overlap
    xB = min(box1[2], box2[2])
    # Lower side of the overlap
    yB = min(box1[3], box2[3])

    # Intersection measures
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area + 1e-6

    return inter_area / union_area


def evaluate_map(predictions: list[dict],
                 targets: list[dict]) -> dict[str, float]:
    """
    Computes mean average precision (mAP) using torchmetrics.

    Source:
    https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html

    Args:
        predictions (list[dict]): Each dict must have:
            - 'boxes' (Tensor[N, 4]): predicted boxes in [x1, y1, x2, y2]
            - 'scores' (Tensor[N]): confidence scores
            - 'labels' (Tensor[N]): predicted class labels
        targets (List[Dict]): Each dict must have:
            - 'boxes' (Tensor[M, 4]): ground truth boxes
            - 'labels' (Tensor[M]): ground truth class labels

    Returns:
        dict[str, float]: Dictionary with mAP and related metrics
    """
    metric = MeanAveragePrecision()
    metric.update(preds=predictions, target=targets)
    result = metric.compute()

    # Convert torch.Tensors to float for readability
    final_result = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                final_result[k] = v.item()
            else:
                final_result[k] = v.tolist()
        else:
            final_result[k] = v

    return final_result
