from typing import List
from .types import LoadedImg
from torchvision import transforms


class AugmentationPipeline:
    """Class to manage a series of augmentations in sequence."""
    def __init__(self, transforms_list: List) -> None:
        """
        Initializes the augmentation pipeline with a list of transformations.
        """
        self._pipeline = transforms.Compose(transforms_list)

    def apply_transforms(self,
                         image: LoadedImg) -> LoadedImg:
        """
        Apply all transformations in the pipeline to the image.

        Args:
            image (LoadedImg): The image to be augmented.

        Returns:
            LoadedImg: The augmented image.
        """
        return self._pipeline(image)