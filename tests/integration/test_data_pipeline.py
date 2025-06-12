import unittest

from src.data.pipeline import DataPreprocessingPipeline
from src.detection_labels import DetectionLabels


class TestDataPipeline(unittest.TestCase):
    def test_pipeline_runs(self):
        pipeline = DataPreprocessingPipeline(DetectionLabels)
        loaders = pipeline.run()
        
        self.assertIn("train", loaders)
        self.assertGreater(len(loaders["train"]), 0)