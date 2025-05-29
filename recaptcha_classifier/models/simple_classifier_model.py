import torch
import torch.nn as nn
import torch.nn.functional as f
from recaptcha_classifier.models.base_model import BaseModel


class SimpleCNN(BaseModel):
    """
    Simple Convolutional Neural Network (CNN) model for image classification.
    """
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 12,
                 conv1_out_channels: int = 6,
                 conv2_out_channels: int = 16,
                 conv1_kernel_size: int = 5,
                 conv2_kernel_size: int = 5,
                 pool_kernel_size: int = 2,
                 pool_stride: int = 2) -> None:
        """
        Initializes the SimpleCNN model.

        Args:
            input_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            conv1_out_channels (int): Number of output channels in the first
            convolutional layer.
            conv2_out_channels (int): Number of output channels in the second
            convolutional layer.
            conv1_kernel_size (int): Kernel size for the first convolutional
            layer.
            conv2_kernel_size (int): Kernel size for the second convolutional
            layer.
            pool_kernel_size (int): Kernel size for the pooling layer.
            pool_stride (int): Stride for the pooling layer.

        Returns:
            None
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv2_kernel_size = conv2_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

        self.conv1 = nn.Conv2d(input_channels, conv1_out_channels,
                               kernel_size=conv1_kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size,
                                 stride=pool_stride)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels,
                               kernel_size=conv2_kernel_size)

        # fixed size after conv and pool layers for input [3, 224, 224]
        flattened_size = 16 * 53 * 53

        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
