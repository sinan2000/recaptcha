import unittest
import numpy as np
from unittest.mock import patch
from PIL import Image
from recaptcha_classifier.data.augment import (
    AugmentationPipeline,
    HorizontalFlip,
    RandomRotation
)


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.img = Image.fromarray(
            np.tile(np.arange(100, dtype=np.uint8).reshape(100, 1),
                    (1, 100, 3))
        )
        self.img = self.img.convert("RGB")

    def test_horizontal_flip(self):
        aug = HorizontalFlip(p=1.0)
        flipped_img = aug.augment(self.img)

        self.assertFalse(np.array_equal(np.array(flipped_img),
                                        np.array(self.img)))

    @patch('random.uniform', return_value=30)
    def test_random_rotation(self, _):
        augmenter = RandomRotation(degrees=30)
        rotated_img = augmenter.augment(self.img)

        self.assertFalse(np.array_equal(np.array(rotated_img),
                                        np.array(self.img)))

    def test_pipeline(self):
        pipeline = AugmentationPipeline()
        pipeline.add_transform(HorizontalFlip(p=1.0))
        pipeline.add_transform(RandomRotation(degrees=30))

        new_img = pipeline.apply_transforms(self.img)

        self.assertIsInstance(new_img, Image.Image)


if __name__ == "__main__":
    unittest.main()
