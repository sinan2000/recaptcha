import random
from abc import ABC, abstractmethod
from PIL import Image
from typing import List
from .types import LoadedImg


class Augmentation(ABC):
    """Abstract class for data augmentation."""
    @abstractmethod
    def augment(self,
                image: Image.Image,
                annotations: List) -> LoadedImg:
        """
        Apply the transformation of the image and updates the bounding boxes
        if necessary.

        Args:
            image (Image.Image): The image to be augmented.
            annotations (List): List of annotations associated with the image.

        Returns:
            LoadedImg: The augmented image and the updated
            annotations.
        """
        pass


class AugmentationPipeline:
    """Class to manage a series of augmentations in sequence."""
    def __init__(self, transforms=[]) -> None:
        self._transforms: List[Augmentation] = transforms

    def apply_transforms(self,
                         image: Image.Image) -> LoadedImg:
        """
        Apply all transformations in the pipeline to the image and
        annotations.

        Args:
            image (Image.Image): The image to be augmented.
            associated with the image.

        Returns:
            LoadedImg: The augmented image and the updated
            annotations.
        """
        for transform in self._transforms:
            if hasattr(transform, 'prob') and random.random() > transform.prob:
                continue
            image, annotations = transform.augment(image)
        return image, annotations


class HorizontalFlip(Augmentation):
    """Flips the image horizontally, with probability p."""
    def __init__(self, p: float = 0.5) -> None:
        self.prob = p

    def augment(self,
                image: Image.Image) -> LoadedImg:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

        return flipped


class RandomRotation(Augmentation):
    """
    Rotates the image by a random angle.
    """
    def __init__(self, degrees: float = 30.0, p: float = 0.5) -> None:
        self._degrees = degrees
        self.prob = p

    def augment(self,
                image: Image.Image) -> LoadedImg:
        angle = random.uniform(-self._degrees, self._degrees)

        rotated = image.rotate(angle)
        return rotated
