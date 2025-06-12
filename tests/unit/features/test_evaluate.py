import unittest
from unittest.mock import MagicMock
import torch
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model


class TestEvaluateModel(unittest.TestCase):
    """
    Unit test for classification model evaluation function.
    """

    def test_evaluate_model_classification(self) -> None:
        """
        Unit test using Magic Mock to simulate a classification model.
        """
        device = torch.device("cpu")

        mock_model = MagicMock()
        # batch_size=2, num_classes=3
        mock_model.return_value = torch.randn(2, 3)

        # Mock DataLoader (one batch with 2 samples and their labels)
        mock_loader = [
            (
                torch.randn(2, 3, 224, 224),   # images
                torch.tensor([0, 1])           # ground truth labels
            )
        ]

        results = evaluate_model(
            model=mock_model,
            test_loader=mock_loader,
            device=device,
            #num_classes=3,
            class_names=["Class0", "Class1", "Class2"]
        )

        # Check expected classification metrics keys
        self.assertIn('Accuracy', results)
        self.assertIn('F1-score', results)

        # Check values are floats
        self.assertIsInstance(results['Accuracy'], float)
        self.assertIsInstance(results['F1-score'], float)


if __name__ == "__main__":
    unittest.main()
