import unittest
from pathlib import Path
from unittest.mock import patch

from recaptcha_classifier.data.pair_loader import ImagePathsLoader


class TestImagePathsLoader(unittest.TestCase):
    @patch("recaptcha_classifier.data.pair_loader.Path.glob")
    @patch("recaptcha_classifier.data.pair_loader.Path.exists",
           return_value=True)
    @patch("recaptcha_classifier.data.pair_loader.Path.is_dir",
           return_value=True)
    def test_load_pairs_with_all_labels(self,
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
            (Path("data/images/class1/img1.png"),
             Path("data/labels/class1/img1.txt")),
            (Path("data/images/class1/img2.png"),
             Path("data/labels/class1/img2.txt")),
        }

        self.assertEqual(set(pairs["class1"]), expected_pairs)
        self.assertEqual(len(pairs["class1"]), 2)

    @patch("recaptcha_classifier.data.pair_loader.Path.glob")
    @patch("recaptcha_classifier.data.pair_loader.Path.exists")
    @patch("recaptcha_classifier.data.pair_loader.Path.is_dir",
           return_value=True)
    def test_load_pairs_with_missing_labels(self,
                                            is_dir_mock,
                                            exists_mock,
                                            glob_mock):
        glob_mock.return_value = [
            Path("data/images/class1/img1.png"),
            Path("data/images/class1/img2.png"),
        ]  # 2 images found in the png glob

        # label folder exists, img1.txt exists but img2.txt does not
        exists_mock.side_effect = [True, True, False]

        loader = ImagePathsLoader(["class1"])
        pairs = loader.find_image_paths()

        self.assertIn("class1", pairs)
        expected_pair = {
            (Path("data/images/class1/img1.png"),
             Path("data/labels/class1/img1.txt")),
        }

        self.assertEqual(set(pairs["class1"]), expected_pair)
        self.assertEqual(len(pairs["class1"]), 1)

    @patch("recaptcha_classifier.data.pair_loader.Path.glob")
    @patch("recaptcha_classifier.data.pair_loader.Path.exists")
    @patch("recaptcha_classifier.data.pair_loader.Path.is_dir")
    def test_caching(self, is_dir_mock, exists_mock, glob_mock):
        loader = ImagePathsLoader(["class1"])
        loader._pairs = {"test": [(Path("test.png"), Path("test.txt"))]}

        # if any mocked method is called, then the cache is not used
        for method in (is_dir_mock, exists_mock, glob_mock):
            method.side_effect = AssertionError(f"{method} called, error!")

        pairs = loader.find_image_paths()

        self.assertIs(pairs, loader._pairs)
        self.assertEqual(pairs, {"test": [(Path("test.png"),
                                           Path("test.txt"))]})


if __name__ == "__main__":
    unittest.main()
