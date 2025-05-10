import math
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


class RandomRotation(Augmentation):
    """
    Rotates the image by a random angle,
    also updates bboxes to reflect the rotation.
    """
    def __init__(self, degrees: float = 30.0) -> None:
        self._degrees = degrees

    def augment(self,
                image: Image.Image,
                annotations: List) -> Tuple[Image.Image, List]:
        angle = random.uniform(-self._degrees, self._degrees)
        rotated = image.rotate(angle)  # check resample param?
        width, height = image.size

        angle_rad = math.radians(angle)
        cen_x, cen_y = width / 2, height / 2
        new_annotations = []

        for ann in annotations:
            x_bb, y_bb, w_bb, h_bb = ann
            # calculate pixel coordinates from bb
            x0 = x_bb * width
            y0 = y_bb * height
            bw = w_bb * width
            bh = h_bb * height

            # also the corner points
            corners = [
                (x0 - bw / 2, y0 - bh / 2),
                (x0 + bw / 2, y0 - bh / 2),
                (x0 + bw / 2, y0 + bh / 2),
                (x0 - bw / 2, y0 + bh / 2)
            ]

            # rotate the corners
            new_corners = []
            for x, y in corners:
                x_rot = (math.cos(angle_rad) * (x - cen_x) -
                         math.sin(angle_rad) * (y - cen_y) + cen_x)
                y_rot = (math.sin(angle_rad) * (x - cen_x) +
                         math.cos(angle_rad) * (y - cen_y) + cen_y)
                new_corners.append((x_rot, y_rot))

            # calculate new bounding box
            x_min = min(x for x, y in new_corners)
            x_max = max(x for x, y in new_corners)
            y_min = min(y for x, y in new_corners)
            y_max = max(y for x, y in new_corners)
            new_x = (x_min + x_max) / (2 * width)
            new_y = (y_min + y_max) / (2 * height)
            new_w = (x_max - x_min) / width
            new_h = (y_max - y_min) / height
            new_annotations.append((new_x, new_y, new_w, new_h))

        return rotated, new_annotations
