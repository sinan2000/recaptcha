import random
from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Tuple


class Augmentation(ABC):
    """Abstract class for data augmentation."""
    @abstractmethod
    def augment(self,
                image: Image.Image,
                annotations: List) -> Tuple[Image.Image, List]:
        """
        Apply the transformation of the image and updates the bounding boxes
        if necessary.

        Args:
            image (Image.Image): The image to be augmented.
            annotations (List): List of annotations associated with the image.

        Returns:
            Tuple[Image.Image, List]: The augmented image and the updated
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
                         annotations: List) -> Tuple[Image.Image, List]:
        """
        Apply all transformations in the pipeline to the image and
        annotations.

        Args:
            image (Image.Image): The image to be augmented.
            annotations (List): List of annotations associated with the image.

        Returns:
            Tuple[Image.Image, List]: The augmented image and the updated
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
                annotations: List) -> Tuple[Image.Image, List]:
        if random.random() > self._p:
            return image, annotations

        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

        # update bounding boxes
        new_annotations = []
        for ann in annotations:
            x, y, w, h = ann
            x2 = 1.0 - x
            new_annotations.append((x2, y, w, h))
        return flipped, new_annotations
