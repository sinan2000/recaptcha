import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import torch
import numpy as np
from recaptcha_classifier.data.dataset import ImageDataset
from recaptcha_classifier.data.preprocessor import ImagePrep
from recaptcha_classifier.data.augment import AugmentationPipeline


class TestImageDataset(unittest.TestCase):
    def setUp(self):
        self.pairs = [(Path("data/images/c1/i1.png"),
                       Path("data/labels/c1/i1.txt"))]
        self.class_map = {"c1": 0}
        self.preprocessor = ImagePrep()
        self.augmentator = AugmentationPipeline()

    @patch.object(ImagePrep, 'load_image')
    @patch.object(ImagePrep, 'load_labels')
    @patch.object(ImagePrep, 'to_tensor')
    def test_loading(self, to_tensor_mock, load_labels_mock, load_image_mock):
        # Mock the return values
        load_image_mock.return_value = MagicMock()
        load_labels_mock.return_value = [(0.1, 0.2, 0.3, 0.4)]
        to_tensor_mock.return_value = torch.rand(3, 224, 224)  # expect shape

        dataset = ImageDataset(
            pairs=self.pairs,
            preprocessor=self.preprocessor,
            augmentator=self.augmentator,
            class_map=self.class_map
        )

        tensor, bboxes, cid = dataset[0]

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertIsInstance(bboxes, list)
        self.assertEqual(bboxes, [(0.1, 0.2, 0.3, 0.4)])
        self.assertIsInstance(cid, int)
        self.assertEqual(cid, 0)

    @patch.object(ImagePrep, 'load_image')
    @patch.object(ImagePrep, 'load_labels', return_value=[])
    def test_empty_bb(self, _, load_image_mock):
        load_image_mock.return_value = MagicMock()
        dataset = ImageDataset(
            pairs=self.pairs,
            preprocessor=self.preprocessor,
            augmentator=self.augmentator,
            class_map=self.class_map
        )
        with self.assertRaises(ValueError):
            dataset[0]

    @patch.object(ImagePrep, 'load_image')
    @patch.object(ImagePrep, 'load_labels',
                  return_value=[(0.1, 0.2, 0.3, 0.4)])
    def test_no_class(self, load_labels_mock, load_image_mock):
        load_image_mock.return_value = np.ones((224, 224, 3))
        dataset = ImageDataset(
            pairs=self.pairs,
            preprocessor=self.preprocessor,
            augmentator=self.augmentator,
        )
        with self.assertRaises(KeyError):
            dataset[0]
