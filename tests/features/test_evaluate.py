import unittest
from unittest.mock import MagicMock
import torch
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model


class TestEvaluateModel(unittest.TestCase):
    """
    Unit test for model evaluation function.
    """

    def test_evaluate_model_with_mocked_model_and_loader(self) -> None:
        """
        Unit test using Magic Mock to simulate a real model.
        """
        device = torch.device("cpu")

        # Mock Model
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.randn(2, 3),              # logits
            torch.rand(2, 4) * 224          # bounding boxes
        )

        # Mock DataLoader (one batch)
        mock_loader = [
            (
                torch.randn(2, 3, 224, 224),   # images
                [
                    {"boxes": torch.tensor([[50., 50., 150., 150.]]),
                     "labels": torch.tensor([0])},
                    {"boxes": torch.tensor([[60., 60., 160., 160.]]),
                     "labels": torch.tensor([1])}
                ]
            )
        ]

        # Run evaluate_model
        results = evaluate_model(
            model=mock_model,
            test_loader=mock_loader,
            device=device,
            num_classes=3,
            eval_classification=True,
            eval_detection=True
        )

        # Assertions
        self.assertIn('Accuracy', results)
        self.assertIn('F1-score', results)
        self.assertIn('map', results)
        self.assertIn('map_50', results)

        self.assertIsInstance(results['Accuracy'], float)
        self.assertIsInstance(results['F1-score'], float)


if __name__ == "__main__":
    unittest.main()
