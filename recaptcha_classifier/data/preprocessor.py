from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np
import torch


class ImagePrep:
    """
    Handles core tasks for handling the data, such as loading, resizing,
    and converting images to tensors. It also parses the YOLO-formated
    label files to extract bounding box annotations.

    It follows the Single Responsibility Principle (SRP) as it separates
    each task, so there is no risk of side effects.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initializes the ImagePrep instance.

        Args:
            target_size (Tuple[int, int]): Desired size for images in (width,
            height) format
        """
        self._target_size: Tuple[int, int] = target_size

    def load_image(self, img_path: Path) -> Image.Image:
        """
        Loads an image from given path, converts it to RGB and resizes it
        to the target size; it is then returned.

        Args:
            img_path (Path): Path to the image file.

        Returns:
            Image.Image: Loaded image in RGB format.
        """
        loaded = Image.open(img_path).convert("RGB")
        return self._resize(loaded)

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

    def class_id_to_tensor(self, c_id: int) -> torch.Tensor:
        """
        Converts a class ID to a tensor.

        Args:
            c_id (int): Class ID to be converted.

        Returns:
            torch.Tensor: Class ID as a tensor.
        """
        return torch.tensor(c_id, dtype=torch.long)
