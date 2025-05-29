import unittest
import torch
from recaptcha_classifier.models.simple_classifier_model import SimpleCNN


class TestSimpleCNNModel(unittest.TestCase):
    """Unit tests for the SimpleCNN model."""

    def setUp(self) -> None:
        """ Set up the test environment. """
        self.model = SimpleCNN()
        self.batch_size = 32
        self.input_shape = (self.batch_size, 3, 224, 224)
        self.mock_input = torch.randn(*self.input_shape)

    def test_forward_output_shape(self) -> None:
        """ Test the output shape of the forward pass. """
        output = self.model(self.mock_input)
        # Expected output shape to be (32,) for 1 int class index per image
        self.assertEqual(output.shape, (self.batch_size,))

    def test_forward_output_type(self) -> None:
        """ Test the output type of the forward pass. """
        output = self.model(self.mock_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.is_floating_point(output) is False)  # No floats

    def test_forward_output_values_are_integers(self) -> None:
        """ Test the output values of the forward pass. """
        output = self.model(self.mock_input)
        self.assertTrue(torch.all(
            output == output.int()), "Output values should be integers")

    def test_model_runs_without_error(self) -> None:
        """ Test that the model runs without error. """
        try:
            _ = self.model(self.mock_input)
        except Exception as e:
            self.fail(f"Model forward pass raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
