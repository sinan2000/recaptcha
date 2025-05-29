import random
from typing import List
from .types import LoadedImg
from torchvision import transforms


class AugmentationPipeline:
    """Class to manage a series of augmentations in sequence."""
    def __init__(self, transforms_list: List) -> None:
        """
        Initializes the augmentation pipeline with a list of transformations.
        """
        self._transforms_list = transforms_list

    def apply_transforms(self,
                         image: LoadedImg) -> LoadedImg:
        """
        Apply a random transformation from the pipeline to the image.

        Args:
            image (LoadedImg): The image to be augmented.

        Returns:
            LoadedImg: The augmented image.
        """
        transform = random.choice(self._transforms_list)
        return transform(image)