import torch
import torch.nn as nn


# It will inherit from Base Class for final draft
class SimpleObjectDetector(nn.Module):
    """
    A simple convolutional neural network for object detection.

    This model predicts bounding box coordinates from an input image.
    It is designed to work with 224x224 RGB images and outputs bounding boxes
    in [x1, y1, x2, y2] pixel coordinate format.

    Args:
        image_size (int): Height and width of the input images
        (default is 224).
    """

    def __init__(self, image_size: int = 224) -> None:
        """
        Initialize the SimpleObjectDetector model.

        Builds the convolutional backbone, flattening layer,
        and output head for bounding box regression.

        Args:
            image_size (int): Expected image dimension (assumes square).
        """
        super(SimpleObjectDetector, self).__init__()

        self.image_size = image_size

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * (image_size // 8) * (image_size // 8), 256),
            nn.ReLU(),
        )

        # Output head for bounding box regression
        self.bbox_head = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, image_size,
            image_size)

        Returns:
            Tensor: Bounding box predictions of shape (batch_size, 4)
                    Format: [x1, y1, x2, y2] in pixel coordinates
        """
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        bbox = self.bbox_head(x)
        return bbox
