import os
import unittest

import torch
from numpy.ma.testutils import assert_equal
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import resnet18
from typing_extensions import overload, override

from recaptcha_classifier.train.training import Trainer


def create_dummy_data(image_size = (3, 32, 32), num_classes = 3, num_samples = 8):
    """
    Creates dummy data for testing purposes.
    :param image_size: tuple with 3 values: channels (RGB=3; greyscale=1), nxn e.g. 32x32 pixels
    :param num_classes: int number of classes
    :param num_samples: Number of images in the dummy dataset
    :return: dummy data
    """

    transform = transforms.ToTensor()  # Converts PIL images to tensors
    ds = FakeData(
        size=num_samples,
        image_size=image_size,
        num_classes=num_classes,
        transform=transform
    )

    #print(ds[0])
    return ds


class TrainingUnitTests(unittest.TestCase):

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_data = create_dummy_data()
        val_data = create_dummy_data(num_samples=2)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=True)
        model = resnet18(weights='DEFAULT')
        model.conv1 = torch.nn.Conv2d(3, 64, 4, (2, 2), (1, 1), bias=False)
        model.fc = torch.nn.Linear(512, 32)
        self.model = model
        optim = torch.optim.RAdam(model.parameters(), lr=0.01)
        self.trainer = Trainer(train_loader=train_loader,
                          val_loader=val_loader,
                          epochs=2,
                          optimizer=optim,
                          scheduler=torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5),
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
        self.trainer.train(model=self.model, load_checkpoint=True)
        assert_equal(os.path.exists(os.path.join(self.trainer.save_folder, self.trainer.model_file_name)),
                     True,
                     "Save folder for model checkpoint doesn't exist.")
        self.trainer.delete_checkpoints()



if __name__ == '__main__':
    unittest.main()
