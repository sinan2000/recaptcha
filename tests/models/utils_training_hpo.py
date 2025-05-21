import torch
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import resnet18


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
    return ds

def initialize_dummy_components(train_data_samples: int, val_data_samples: int, batch_size: int, in_channels: int):
    train_data = create_dummy_data(num_samples=train_data_samples)
    val_data = create_dummy_data(num_samples=val_data_samples)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    model = resnet18(weights='DEFAULT')
    model.conv1 = torch.nn.Conv2d(in_channels, 64, 4, (2, 2), (1, 1), bias=False)
    model.fc = torch.nn.Linear(512, 32)
    optim = torch.optim.RAdam(model.parameters(), lr=0.01)
    return model, optim, train_loader, val_loader