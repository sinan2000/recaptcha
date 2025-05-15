import unittest
from unittest.mock import patch
from recaptcha_classifier.data.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    def setUp(self):
        # only written pairs as string, instead of paths, for simplicity
        self.sample_splits = {
            "train": {
                "class1": [("img1", "label1"), ("img2", "label2")],
                "class2": [("img3", "label3"), ("img4", "label4")],
            },
            "val": {
                "class1": [("img5", "label5")],
                "class2": [("img6", "label6")],
            },
            "test": {
                "class1": [("img7", "label7")],
                "class2": [("img8", "label8")],
            },
        }

    def test_print_counts(self):
        with patch('builtins.print') as print_mock:
            Visualizer.print_counts(self.sample_splits)
            print_mock.assert_any_call("TRAIN:")
            print_mock.assert_any_call("  class1: 2")
            print_mock.assert_any_call("  class2: 2")
            print_mock.assert_any_call("VAL:")
            print_mock.assert_any_call("  class1: 1")
            print_mock.assert_any_call("  class2: 1")
            print_mock.assert_any_call("TEST:")
            print_mock.assert_any_call("  class1: 1")
            print_mock.assert_any_call("  class2: 1")

    @patch("recaptcha_classifier.data.visualizer.plt.show")
    def test_plot_splits(self, show_mock):
        Visualizer.plot_splits(self.sample_splits)
        show_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
