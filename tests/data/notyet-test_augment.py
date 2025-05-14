import unittest
import numpy as np
from unittest.mock import patch
from PIL import Image
from recaptcha_classifier.data.augment import (
    AugmentationPipeline,
    HorizontalFlip,
    RandomRotation
)
from recaptcha_classifier.data.scaler import YOLOScaler


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.img = Image.fromarray(
            np.tile(np.arange(100, dtype=np.uint8).reshape(100, 1),
                    (1, 100, 3))
        )
        self.img = self.img.convert("RGB")
        self.bb = [(0.5, 0.5, 0.2, 0.2)]

    def test_horizontal_flip(self):
        aug = HorizontalFlip(p=1.0)
        flipped_img, flipped_bb = aug.augment(self.img, self.bb)

        self.assertFalse(np.array_equal(np.array(flipped_img),
                                        np.array(self.img)))

        result = YOLOScaler.scale_for_flip(self.bb)
        self.assertEqual(flipped_bb, result)

    @patch('random.uniform', return_value=30)
    def test_random_rotation(self, _):
        augmenter = RandomRotation(degrees=30)
        rotated_img, rotated_bb = augmenter.augment(self.img, self.bb)

        self.assertFalse(np.array_equal(np.array(rotated_img),
                                        np.array(self.img)))

        result = YOLOScaler.scale_for_rotation(self.bb, 30, self.img.size)
        for i, j in zip(rotated_bb, result):
            self.assertAlmostEqual(i, j, places=4)

    def test_pipeline(self):
        pipeline = AugmentationPipeline()
        pipeline.add_transform(HorizontalFlip(p=1.0))
        pipeline.add_transform(RandomRotation(degrees=30))

        new_img, new_bb = pipeline.apply_transforms(self.img, self.bb)

        self.assertIsInstance(new_img, Image.Image)
        self.assertIsInstance(new_bb, list)


if __name__ == "__main__":
    unittest.main()
