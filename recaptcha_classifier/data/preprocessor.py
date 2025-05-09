from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch


class Preprocessor:
    """
    Responsible for all image-label preprocessing steps:
    1. Load images from disk
    2. Parse YOLO label files
    3. Resize images (we keep same square aspect ratio*, therefore no
    need to modify YOLO labels in our case)
    4. Normalize pixel values to [0, 1]

    We previously confirmed that all images in our dataset are square,
    of 120x120 pixels. Moreover, as they have such small size, and we
    totally have around 500 images, we decided to use pillow instead of
    its faster alternative, openCV, for the sake of simplicity.

    Follows Single Responsibility Principle (SRP), as only data loading &
    transformation is handled.
    Uses Template Method Pattern, because process_pair() defines the
    skeleton of the algorithm, with detailed steps implemented in protected
    private methods.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initializes the Preprocessor instance.

        Args:
            target_size (Tuple[int, int]): Desired size for images in (width,
            height) format
        """
        self._target_size: Tuple[int, int] = target_size

    def process_pairs(self,
                      img_path: Path,
                      lbl_path: Path) -> Tuple[torch.Tensor, List[float]]:
        """
        Public method to load and then preprocess an image, also parsing
        its corresponding YOLO label file.

        Args:
            img_path (Path): Path to the image file.
            lbl_path (Path): Path to the label file.

        Returns:
            - image tensor of shape (3, H, W) with normalized pixel values
            between 0 and 1
            - list of annotatation bounding boxes in YOLO format
            (x_center, y_center, width, height)
        """
        img = self._load_image(img_path)
        img = self._resize(img)
        tensor = self._to_tensor(img)
        bbox = self._load_labels(lbl_path)
        return tensor, bbox

    def _load_image(self, img_path: Path) -> Image.Image:
        """
        Private helper method to open an image and convert it to RGB.

        Args:
            img_path (Path): Path to the image file.

        Returns:
            Image.Image: Loaded image in RGB format.
        """
        return Image.open(img_path).convert("RGB")

    def _resize(self, img: Image.Image) -> Image.Image:
        """
        Private helper method to resize an image to the target size.

        Args:
            img (Image.Image): Image to be resized.

        Returns:
            Image.Image: Resized image.
        """
        return img.resize(self._target_size, Image.LANCZOS)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        Private helper to convert received PIL image into a PyTorch tensor.
        It also normalizes the pixel values to [0, 1].

        Args:
            img (Image.Image): Image to be converted.

        Returns:
            torch.Tensor: Converted image tensor with pixel values in [0, 1].
        """
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def _load_labels(self, lbl_path: Path) -> List[float]:
        """
        Private helper method to parse one YOLO label file.

        Args:
            lbl_path (Path): Path to the label file.

        Returns:
            List[float]: List of bounding boxes in YOLO format
            (x_center, y_center, width, height).
        """
        bounding_boxes: List[float] = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # invalid line skipped
                _, x_center, y_center, width, height = map(float, parts)
                bounding_boxes.append((x_center, y_center, width, height))

        return bounding_boxes
