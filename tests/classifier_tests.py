import unittest
import torch
from recaptca_classifier.models.simple_classifier_model import Net


class TestNetModel(unittest.TestCase):

    def test_forward_output_shape(self):
        model = Net()
        mock_input = torch.randn(4, 3, 32, 32) # generate 4 random rgb mock images with size 32x32
        output = model(mock_input)
        self.assertEqual(output.shape, (4, 10))  # 4 images, 10 class scores each

    def test_forward_output_type(self):
        model = Net()
        mock_input = torch.randn(4, 3, 32, 32)
        output = model(mock_input)
        self.assertIsInstance(output, torch.Tensor)

    def test_model_runs_without_error(self):
        model = Net()
        mock_input = torch.randn(4, 3, 32, 32)
        try:
            _ = model(mock_input)
        except Exception as e:
            self.fail(f"Model forward pass raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()