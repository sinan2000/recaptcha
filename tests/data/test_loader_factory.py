import unittest
from pathlib import Path
from unittest.mock import patch
from recaptcha_classifier.data.loader_factory import LoaderFactory
from recaptcha_classifier.data.preprocessor import ImagePrep
from recaptcha_classifier.data.augment import AugmentationPipeline
from torch.utils.data import DataLoader


class TestLoaderFactory(unittest.TestCase):
    @patch("recaptcha_classifier.data.loader_factory.ImageDataset")
    def test_create_loaders(self, dataset_mock):
        class_map = {"class1": 0, "class2": 1}
        preprocessor = ImagePrep()
        augmentator = AugmentationPipeline()
        factory = LoaderFactory(class_map, preprocessor, augmentator)

        dataset_mock.return_value = dataset_mock
        dataset_mock.__len__.return_value = 7  # total number of items

        splits = {
            "train": {
                "class1": [(Path("img1.png"), Path("label1.txt")),
                           (Path("img2.png"), Path("label2.txt"))],
                "class2": [(Path("img3.png"), Path("label3.txt"))]
            },
            "val": {
                "class1": [(Path("img4.png"), Path("label4.txt"))],
                "class2": [(Path("img5.png"), Path("label5.txt"))]
            },
            "test": {
                "class1": [(Path("img6.png"), Path("label6.txt"))],
                "class2": [(Path("img7.png"), Path("label7.txt"))]
            }
        }

        loaders = factory.create_loaders(splits)

        self.assertIn("train", loaders)
        self.assertIn("val", loaders)
        self.assertIn("test", loaders)

        for loader in loaders.values():
            self.assertIsInstance(loader, DataLoader)

        # We also need to check that the train set has the augmentator
        dataset_mock.assert_any_call(
            pairs=[(Path("img1.png"), Path("label1.txt")),
                   (Path("img2.png"), Path("label2.txt")),
                   (Path("img3.png"), Path("label3.txt"))],
            preprocessor=preprocessor,
            augmentator=augmentator,
            class_map=class_map
        )


if __name__ == "__main__":
    unittest.main()
