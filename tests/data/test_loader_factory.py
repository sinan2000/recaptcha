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
        augmentator = AugmentationPipeline(transforms_list=[])
        factory = LoaderFactory(class_map, preprocessor, augmentator)

        dataset_mock.return_value = dataset_mock
        dataset_mock.__len__.return_value = 7  # total number of items

        splits = {
            "train": {
                "class1": [Path("img1.png"), Path("img2.png")],
            },
            "val": {
                "class1":
                    [Path("img3.png")],
            },
            "test": {
                "class1": [Path("img4.png"), Path("img5.png"), Path("img6.png")]
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
            items=[Path("img1.png"),Path("img2.png")],
            preprocessor=preprocessor,
            augmentator=augmentator,
            class_map=class_map
        )


if __name__ == "__main__":
    unittest.main()
