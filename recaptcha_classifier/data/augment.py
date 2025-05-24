import random
from abc import ABC, abstractmethod
from PIL import Image
from typing import List
from .scaler import YOLOScaler
from .types import LoadedImg, BBoxList


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
                         image: Image.Image,
                         annotations: BBoxList) -> LoadedImg:
        """
        Apply all transformations in the pipeline to the image and
        annotations.

        Args:
            image (Image.Image): The image to be augmented.
            annotations (BBoxList): List of annotations
            associated with the image.

        Returns:
            LoadedImg: The augmented image and the updated
            annotations.
        """
        for transform in self._transforms:
            if hasattr(transform, 'prob') and random.random() > transform.prob:
                continue
            image, annotations = transform.augment(image, annotations)
        return image, annotations


class HorizontalFlip(Augmentation):
    """Flips the image horizontally, with probability p and updates bboxes."""
    def __init__(self, p: float = 0.5) -> None:
        self.prob = p

    def augment(self,
                image: Image.Image,
                annotations: BBoxList) -> LoadedImg:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_annotations = YOLOScaler.scale_for_flip(annotations)

        return flipped, new_annotations


class RandomRotation(Augmentation):
    """
    Rotates the image by a random angle,
    also updates bboxes to reflect the rotation.
    """
    def __init__(self, degrees: float = 30.0, p: float = 0.5) -> None:
        self._degrees = degrees
        self.prob = p

    def augment(self,
                image: Image.Image,
                annotations: BBoxList) -> LoadedImg:
        angle = random.uniform(-self._degrees, self._degrees)

        rotated = image.rotate(angle)
        new_annotations = (YOLOScaler
                           .scale_for_rotation(annotations,
                                               angle,
                                               image.size)
                           )

        return rotated, new_annotations
