import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.paths_loader import ImagePathsLoader


class TestImagePathsLoader(unittest.TestCase):
    @patch("src.data.paths_loader.Path.glob")
    @patch("src.data.paths_loader.Path.exists",
           return_value=True)
    @patch("src.data.paths_loader.Path.is_dir",
           return_value=True)
    def test_load_paths(self,
                                        is_dir_mock,
                                        exists_mock,
                                        glob_mock):
        glob_mock.return_value = [
            Path("data/images/class1/img1.png"),
            Path("data/images/class1/img2.png"),
        ]  # 2 images found in the png glob

        loader = ImagePathsLoader(["class1"])
        pairs = loader.find_image_paths()

        expected_pairs = {
            Path("data/images/class1/img1.png"),
            Path("data/images/class1/img2.png"),
        }

        self.assertEqual(set(pairs["class1"]), expected_pairs)
        self.assertEqual(len(pairs["class1"]), 2)

if __name__ == "__main__":
    unittest.main()
