import unittest
import torch
import src.features.evaluation.classification_metrics as cm


class TestEvaluateClassification(unittest.TestCase):
    """
    Unit tests for classification metrics like Accuracy, F1 score
    and confusion matrix.
    """

    def test_perfect_classification(self) -> None:
        """
        Unit test in case of a perfect prediction.
        """
        y_true = torch.tensor([0, 1, 2])
        y_logits = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = cm.evaluate_classification(y_logits, y_true, num_classes=3,
                                            cm_plot=False)

        self.assertAlmostEqual(result['Accuracy'], 1.0, places=6)
        self.assertAlmostEqual(result['F1-score'], 1.0, places=6)
        self.assertIsInstance(result['Confusion Matrix'], torch.Tensor)
        self.assertEqual(result['Confusion Matrix'].shape, (3, 3))

    def test_wrong_predictions(self) -> None:
        """
        Unit test in case of wrong predictions for all labels.
        """
        y_true = torch.tensor([0, 1, 2])
        y_logits = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        result = cm.evaluate_classification(y_logits, y_true, num_classes=3,
                                            cm_plot=False)

        self.assertEqual(result['Accuracy'], 0.0)
        self.assertAlmostEqual(result['F1-score'], 0.0, places=6)

    def test_partial_predictions(self) -> None:
        """
        Unit test in case of partial accuracy of predictions.
        """
        y_true = torch.tensor([0, 1, 2, 1])
        y_logits = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.3, 0.6, 0.1],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
        ])
        result = cm.evaluate_classification(y_logits, y_true, num_classes=3,
                                            cm_plot=False)

        self.assertGreater(result['Accuracy'], 0.0)
        self.assertLess(result['Accuracy'], 1.0)
        self.assertGreater(result['F1-score'], 0.0)
        self.assertLess(result['F1-score'], 1.0)
        self.assertEqual(result['Confusion Matrix'].shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
