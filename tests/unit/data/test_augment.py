import unittest
import numpy as np
from unittest.mock import patch
from PIL import Image
from src.data.augment import AugmentationPipeline
from torchvision import transforms


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.img = Image.fromarray(
            np.tile(np.arange(100, dtype=np.uint8).reshape(100, 1),
                    (1, 100, 3))
        )
        self.img = self.img.convert("RGB")

    def test_pipeline(self):
        pipeline = AugmentationPipeline([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=30)
        ])

        new_img = pipeline.apply_transforms(self.img)

        self.assertIsInstance(new_img, Image.Image)


if __name__ == "__main__":
    unittest.main()
