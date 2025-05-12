import unittest
import torch
import recaptcha_classifier.models.simple_object_detector as od


class TestSimpleObjectDetector(unittest.TestCase):
    """
    Unit test for the simple object detection class.
    """

    def test_forward_output_shape(self) -> None:
        """
        Test that the model outputs a bounding box tensor of
        shape (batch_size, 4)
        """
        model = od.SimpleObjectDetector()
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        self.assertEqual(output.shape, (2, 4), "Output shape should be"
                         " (batch_size, 4)")

    def test_output_values_are_finite(self) -> None:
        """
        Test that the bounding box values are finite (no NaN or Inf)
        """
        model = od.SimpleObjectDetector()
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        self.assertTrue(torch.isfinite(output).all(), "Bounding box values"
                        " should be finite")


if __name__ == "__main__":
    unittest.main()
