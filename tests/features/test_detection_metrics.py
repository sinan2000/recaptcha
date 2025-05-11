import unittest
import torch
from recaptcha_classifier.features.evaluation.detection_metrics import (
    compute_iou,
    yolo_to_corners,
    evaluate_map,
)


class TestEvaluateDetection(unittest.TestCase):
    """
    Unit tests for the object detection metrics like IoU and mAP.
    """

    def test_iou_perfect_overlap(self) -> None:
        """
        Unit test for IoU in the case in which there is complete overlap
        between the predicted box and the ground truth box.
        """
        box = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_iou(box, box), 1.0, places=6)

    def test_iou_no_overlap(self) -> None:
        """
        Unit test for IoU in the case in which there is no overlap
        between the predicted box and the ground truth box.
        """
        box1 = [10, 10, 50, 50]
        box2 = [60, 60, 100, 100]
        self.assertEqual(compute_iou(box1, box2), 0.0)

    def test_iou_partial_overlap(self) -> None:
        """
        Unit test for IoU in the case in which there is partial overlap
        between the predicted box and the ground truth box.
        """
        box1 = [10, 10, 60, 60]
        box2 = [30, 30, 80, 80]
        iou = compute_iou(box1, box2)
        self.assertTrue(0 < iou < 1)

    def test_yolo_to_corners_centered_square(self) -> None:
        """
        Unit test for the conversion function yolo_to_corners.
        """
        img_w, img_h = 100, 100
        result = yolo_to_corners(0.5, 0.5, 0.2, 0.2, img_w, img_h)
        expected = [40.0, 40.0, 60.0, 60.0]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b, places=6)

    def test_evaluate_map_single_perfect(self) -> None:
        """
        Unit test for mAP metric, in case of perfect prediction.
        """
        box = [10, 10, 50, 50]
        preds = [{
            "boxes": torch.tensor([box]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0])
        }]
        targets = [{
            "boxes": torch.tensor([box]),
            "labels": torch.tensor([0])
        }]
        result = evaluate_map(preds, targets)
        self.assertEqual(result["map_50"], 1.0)
        self.assertGreaterEqual(result["map"], 0.5)


if __name__ == "__main__":
    unittest.main()
