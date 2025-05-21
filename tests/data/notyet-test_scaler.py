import unittest
from recaptcha_classifier.data.scaler import YOLOScaler


class TestYOLOScaler(unittest.TestCase):
    def test_scale_for_flip(self):
        bb = [(0.5, 0.5, 0.2, 0.2), (0.3, 0.4, 0.1, 0.1)]
        flipped = YOLOScaler.scale_for_flip(bb)
        self.assertEqual(flipped, [(0.5, 0.5, 0.2, 0.2),
                                   (0.7, 0.4, 0.1, 0.1)])

    def test_scale_for_rotation(self):
        bb = [(0.5, 0.5, 0.4, 0.2)]
        rot = YOLOScaler.scale_for_rotation(bb, 90, (224, 224))
        self.assertEqual(len(rot), 1)
        self.assertTrue(all(0 <= v <= 1 for bb in rot for v in bb))

    def test_empty_list(self):
        self.assertEqual(YOLOScaler.scale_for_flip([]), [])
        self.assertEqual(YOLOScaler.scale_for_rotation([], 45, (224, 224)), [])
