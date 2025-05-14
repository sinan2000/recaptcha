import unittest
import torch
from torch import nn
from torchvision import transforms
from recaptcha_classifier.features.models.simple_classifier_model import Net

class TestNetModel(unittest.TestCase):
    def setUp(self):
        self.model = Net()
        self.model.eval()  # Set model to evaluation mode

        # Create a mock batch of 4 RGB images, each of size 32x32
        self.mock_input = torch.randn(4, 3, 32, 32)

    def test_forward_output_shape(self):
        output = self.model(self.mock_input)
        # The model should output 4 predictions with 10 class scores each
        self.assertEqual(output.shape, (4, 10))

    def test_forward_output_type(self):
        output = self.model(self.mock_input)
        self.assertIsInstance(output, torch.Tensor)

    def test_model_runs_without_error(self):
        try:
            _ = self.model(self.mock_input)
        except Exception as e:
            self.fail(f"Model forward pass raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
