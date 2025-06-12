import unittest
from src.data.splitter import DataSplitter


class TestDataSplitter(unittest.TestCase):
    def test_default_split(self):
        data = {"test_class": [f"img_{i}" for i in range(10)]}
        splits = DataSplitter().split(data)

        self.assertEqual(len(splits['train']['test_class']), 7)
        self.assertEqual(len(splits['val']['test_class']), 2)
        self.assertEqual(len(splits['test']['test_class']), 1)

    def test_zero_split(self):
        data = {"test_class": [(f"img_{i}", f"label_{i}") for i in range(10)]}
        splits = DataSplitter(ratios=(0.0, 1.0, 0.0)).split(data)

        self.assertEqual(len(splits['train']['test_class']), 0)
        self.assertEqual(len(splits['val']['test_class']), 10)
        self.assertEqual(len(splits['test']['test_class']), 0)

    def test_wrong_ratios(self):
        with self.assertRaises(ValueError):
            DataSplitter(ratios=(1, 1, 1, 1))  # not 3 ratios
        with self.assertRaises(ValueError):
            DataSplitter(ratios=(1, 1, 1))  # sum bigger than 1
        with self.assertRaises(ValueError):
            DataSplitter(ratios=(1.1, -0.1, 0))  # negative ratio

    def test_reproducibility(self):
        data = {"test_class": [(f"img_{i}", f"label_{i}") for i in range(10)]}
        splitter1 = DataSplitter(seed=42)
        splitter2 = DataSplitter(seed=42)

        splits1 = splitter1.split(data)
        splits2 = splitter2.split(data)

        self.assertEqual(splits1, splits2)


if __name__ == '__main__':
    unittest.main()
