from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np
import torch
from .types import BBoxList


class Preprocessor:
    """
    Handles core tasks for handling the data, such as loading, resizing,
    and converting images to tensors. It also parses the YOLO-formated
    label files to extract bounding box annotations.

    It follows the Single Responsibility Principle (SRP) as it separates
    each task, so there is no risk of side effects.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initializes the Preprocessor instance.

        Args:
            target_size (Tuple[int, int]): Desired size for images in (width,
            height) format
        """
        self._target_size: Tuple[int, int] = target_size

    def load_image(self, img_path: Path) -> Image.Image:
        """
        Loads an image from given path and converts it to RGB.
        Also resizes the image to the target size.

        Args:
            img_path (Path): Path to the image file.

        Returns:
            Image.Image: Loaded image in RGB format.
        """
        loaded = Image.open(img_path).convert("RGB")
        return self._resize(loaded)

    def load_labels(self, lbl_path: Path) -> BBoxList:
        """
        Parses the list of YOLO format labels from the file at the given path.

        Args:
            lbl_path (Path): Path to the label file.

        Returns:
            BBoxList: List of bounding boxes in YOLO format
            (x_center, y_center, width, height).
        """
        bounding_boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # invalid line skipped
                _, x_center, y_center, width, height = map(float, parts)
                bounding_boxes.append((x_center, y_center, width, height))

        return bounding_boxes

    def _resize(self, img: Image.Image) -> Image.Image:
        """
        Resizes the image to the target size.

        Args:
            img (Image.Image): Image to be resized.

        Returns:
            Image.Image: Resized image.
        """
        return img.resize(self._target_size, Image.LANCZOS)

    def to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        Converts given image into a normalized PyTorch tensor.

        Args:
            img (Image.Image): Image to be converted.

        Returns:
            torch.Tensor: Normalized image tensor
        """
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)
