import unittest
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
from recaptcha_classifier.data.preprocessor import ImagePrep


class TestImagePrep(unittest.TestCase):
    def setUp(self):
        self.prep = ImagePrep()

    @patch("PIL.Image.open")
    def test_load_image(self, open_mock):
        img_mock = MagicMock(Image.Image)
        resized_mock = MagicMock(Image.Image)

        img_mock.convert.return_value = resized_mock
        open_mock.return_value = img_mock

        self.prep.load_image(Path("test"))

        open_mock.assert_called_once_with(Path("test"))
        img_mock.convert.assert_called_once_with("RGB")
        resized_mock.resize.assert_called_once_with((224, 224), Image.LANCZOS)

    @patch("builtins.open", new_callable=unittest.mock.mock_open,
           read_data="0 0.2 0.3 0.4 0.5\n")
    def test_load_labels(self, file_mock):
        result = self.prep.load_labels(Path("test"))

        self.assertEqual(result, [(0.2, 0.3, 0.4, 0.5)])

    def test_to_tensor(self):
        img = Image.new("RGB", (224, 224))
        tensor = self.prep.to_tensor(img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertTrue(torch.all(tensor.ge(0) & tensor.le(1)))


if __name__ == "__main__":
    unittest.main()
