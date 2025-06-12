import torch
import torch
import torch.nn as nn
from src.constants import INPUT_SHAPE
from src.models.base_model import BaseModel


class ResidualBlock(nn.Module):
    """
    Residual convolutional block that has 2 conv layers,
    inspired by ResNet architecture.
    We use this to allow a better gradient flow.

    The implementation details can be found at:
    https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        """
        Initializes the ResidualBlock with two convolutional layers,
        batch normalization, and ReLU activation.
        
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolutional kernel.
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        :param x: Input tensor.
        :return: Output tensor.
        """
        identity = self.downsample(x) if self.downsample else x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)
    

class MainCNN(BaseModel):
    """
    Main CNN model, that now uses Residual Blocks.
    """
    def __init__(self,
                 n_layers: int,
                 kernel_size: int,
                 num_classes: int = 12,
                 input_shape: tuple = INPUT_SHAPE,
                 base_channels: int = 32) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        curr_channels = self.input_shape[0]
        self.res_blocks = nn.ModuleList()
        
        for l_idx in range(n_layers):
            out_channels = min(base_channels * (2 ** l_idx), 128)
            stride = 2 if l_idx > 0 else 1 # to increase training speed
            self.res_blocks.append(
                ResidualBlock(curr_channels, out_channels, kernel_size, stride)
            )
            curr_channels = out_channels

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(curr_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        for block in self.res_blocks:
            x = block(x)
        
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x