import unittest

from torchvision.models import resnet18

from recaptcha_classifier.train.training import Trainer


class TestHPOptimizer(unittest.TestCase):

    def __init__(self):
        super(TestHPOptimizer, self).__init__()
        model = resnet18(weights='DEFAULT')
        trainer = Trainer()

    def test_generating_combos(self):
        self.assertEqual(True, False)  # add assertion here

    def test_train_one_model(self):
        pass

    def test_retrieve_results(self):
        pass

if __name__ == '__main__':
    unittest.main()
