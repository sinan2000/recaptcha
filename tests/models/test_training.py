import os
import unittest

import numpy as np
import torch
from numpy.ma.testutils import assert_equal
from torchvision.models import resnet18

from recaptcha_classifier.train.training import Trainer


def _create_dummy_data(shape: list) -> torch.DataFrame:
    arr = np.random.rand(*shape)
    # flatten the array so it fits into a DataFrame structure (samples, features)
    flat_arr = arr.reshape(shape[0], -1)
    # convert to dataframe
    df = torch.DataFrame(flat_arr)
    return df


class TrainingUnitTests(unittest.TestCase):

    def _initialize(self) -> tuple:
        data = _create_dummy_data([3, 224, 224])
        trainloader = torch.utils.data.DataLoader(data[:8], batch_size=2, shuffle=True)
        val_loader = torch.utils.data.DataLoader(data[8:], batch_size=2, shuffle=True)
        model = resnet18(weights='DEFAULT')
        # change params
        model.conv1 = torch.nn.Conv2d(3, 64, 4, (2, 2), (3, 3), bias=False)  # ?
        model.fc = torch.nn.Linear(512, 32)
        optim = torch.optim.RAdam(model.parameters(), lr=0.01)
        trainer = Trainer(train_loader=trainloader, val_loader=val_loader,
                          epochs=2,
                          optimizer=optim,
                          scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim),
                          save_folder='test_training_checkpoints')

        return model, trainer


    def test_device(self):
        trainer = self._initialize()[3]
        assert trainer.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def test_train_process(self) -> None:
        model, trainer = self._initialize()
        param1 = model.parameters()
        trainer.train(model=model, load_checkpoint=False)
        param2 = model.parameters()
        assert_equal(param1 == param2, False,
                     "Model parameters have not changed. No training occurred. ")
        trainer.delete_checkpoints()


    def test_train_load_checkpoint(self):
        model, trainer = self._initialize()
        trainer.train(model=model, load_checkpoint=True)
        assert_equal(os.path.exists(os.path.join(trainer.save_folder, trainer.model_file_name)),
                     True,
                     "Save folder for model checkpoint doesn't exist.")
        trainer.delete_checkpoints()



if __name__ == '__main__':
    unittest.main()
