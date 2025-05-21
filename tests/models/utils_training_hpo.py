import torch
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import resnet18


def create_dummy_data(image_size = (3, 32, 32), num_classes = 3, num_samples = 8):
    """
    Creates fake dummy data for testing purposes.
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
    return ds


def initialize_dummy_components(train_data_samples: int, val_data_samples: int, batch_size: int, num_classes: int):
    """
    Creates: train and validation data_loaders with random data;
    a dummy resnet18 model;
    a RAdam optimizer with learning rate 0.01.
    :param train_data_samples: number of training data samples to be generated
    :param val_data_samples: number of validation data samples to be generated
    :param batch_size: batch size
    :param num_classes: number of classes in the dummy dataset
    :return: model, optimizer, train_loader, val_loader
    """
    train_data = create_dummy_data(num_classes=num_classes, num_samples=train_data_samples)
    val_data = create_dummy_data(num_classes=num_classes, num_samples=val_data_samples)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    model = resnet18(weights='DEFAULT')
    model.conv1 = torch.nn.Conv2d(3, 64, 4, (2, 2), (1, 1), bias=False)
    model.fc = torch.nn.Linear(512, 32)
    optim = torch.optim.RAdam(model.parameters(), lr=0.01)
    return model, optim, train_loader, val_loader
