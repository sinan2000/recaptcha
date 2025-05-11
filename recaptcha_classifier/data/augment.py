import random
from abc import ABC, abstractmethod
from PIL import Image
from typing import List
from .bbox_scaler import BoundingBoxScaler
from .types import DataPair, BBoxList


class Augmentation(ABC):
    """Abstract class for data augmentation."""
    @abstractmethod
    def augment(self,
                image: Image.Image,
                annotations: List) -> DataPair:
        """
        Apply the transformation of the image and updates the bounding boxes
        if necessary.

        Args:
            image (Image.Image): The image to be augmented.
            annotations (List): List of annotations associated with the image.

        Returns:
            DataPair: The augmented image and the updated
            annotations.
        """
        pass


class AugmentationPipeline:
    """Class to manage a series of augmentations in sequence."""
    def __init__(self) -> None:
        self._transforms: List[Augmentation] = []

    def add_transform(self, transform: Augmentation) -> None:
        """
        Add a new transformation to the pipeline.

        Args:
            transform (Augmentation): The augmentation to be added.
        """
        self._transforms.append(transform)

    def apply_transforms(self,
                         image: Image.Image,
                         annotations: BBoxList) -> DataPair:
        """
        Apply all transformations in the pipeline to the image and
        annotations.

        Args:
            image (Image.Image): The image to be augmented.
            annotations (BBoxList): List of annotations
            associated with the image.

        Returns:
            DataPair: The augmented image and the updated
            annotations.
        """
        for transform in self._transforms:
            image, annotations = transform.augment(image, annotations)
        return image, annotations


class HorizontalFlip(Augmentation):
    """Flips the image horizontally, with probability p and updates bboxes."""
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def augment(self,
                image: Image.Image,
                annotations: BBoxList) -> DataPair:
        if random.random() > self._p:
            return image, annotations

        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_annotations = BoundingBoxScaler.scale_for_flip(annotations)

        return flipped, new_annotations


class RandomRotation(Augmentation):
    """
    Rotates the image by a random angle,
    also updates bboxes to reflect the rotation.
    """
    def __init__(self, degrees: float = 30.0) -> None:
        self._degrees = degrees

    def augment(self,
                image: Image.Image,
                annotations: BBoxList) -> DataPair:
        angle = random.uniform(-self._degrees, self._degrees)

        rotated = image.rotate(angle)
        new_annotations = (BoundingBoxScaler
                           .scale_for_rotation(annotations,
                                               angle,
                                               image.size)
                           )

        return rotated, new_annotations
