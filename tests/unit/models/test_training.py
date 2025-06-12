import os
import unittest
import torch
from numpy.ma.testutils import assert_equal
from typing_extensions import override

from src.train.training import Trainer
from tests.models.utils_training_hpo import initialize_dummy_components

import torch
from numpy.ma.testutils import assert_equal
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import resnet18
from typing_extensions import overload, override

from src.train.training import Trainer


class TrainingUnitTests(unittest.TestCase):

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model, train_loader, val_loader = initialize_dummy_components(8,2,2, 3)

        self.model = model

        self.trainer = Trainer(train_loader=train_loader,
                          val_loader=val_loader,
                          epochs=2,
                          save_folder='test_training_checkpoints')


    def test_device(self):
        assert self.trainer.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def test_train_process(self) -> None:
        param1 = self.model.parameters()
        self.trainer.train(model=self.model, load_checkpoint=False)
        param2 = self.model.parameters()
        assert_equal(param1 == param2, False,
                     "Model parameters have not changed. No training occurred.")
        self.trainer.delete_checkpoints()


    def test_train_load_checkpoint(self):
        self.trainer.train(model=self.model, load_checkpoint=False)
        self.trainer.train(model=self.model, load_checkpoint=True)
        assert_equal(os.path.exists(os.path.join(self.trainer.save_folder, self.trainer.model_file_name)),
                     True,
                     "Save folder for model checkpoint doesn't exist.")
        self.trainer.delete_checkpoints()



if __name__ == '__main__':
    unittest.main()
