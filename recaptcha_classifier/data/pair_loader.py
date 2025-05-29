import logging
from pathlib import Path
from typing import List, Dict
from .types import ClassFileDict

logger = logging.getLogger(__name__)


class ImageLabelLoader:
    """
    This class loads all image-label pairs for given classes.

    It scans for all matching images and labels and caches the result.

    It follows Single Responsibility Principle (SRP) as it only handles
    the loading of the pairs. Also, it uses the Iterator pattern, as
    it can be looped over to get the list pairs as tuples by class,
    in format (class, [(img_path, lbl_path), ...].)
    """
    def __init__(self,
                 classes: List[str],
                 images_dir: str = "data/images"
                 ) -> None:
        """
        Initializes the PairsLoader instance.

        Args:
            classes (List[str]): List of class names to load.
            images_dir (str): Path to the directory containing images.

        Returns:
            None
        """
        self._classes = classes
        self._images_dir = Path(images_dir)
        self._pairs: ClassFileDict = dict()

    def find_pairs(self) -> ClassFileDict:
        """
        Returns all pairs loaded for the given classes.
        It caches the response after first run.

        Returns:
            List[Tuple[Path, Path]]: List of tuples
            containing image and label paths. (img_path, lbl_path)
        """
        if not self._pairs:
            self._load_pairs()
        return self._pairs

    def __iter__(self) -> None:
        """
        Iterates over classes and their respective list of pairs.

        Yields:
            Tuple[str, List[Tuple[Path, Path]]]: Class name and list of
            tuples containing image and label paths.
        """
        for cls, pairs in self.find_pairs().items():
            yield cls, pairs

    def __len__(self) -> int:
        """
        Returns total number of matched pairs.

        Returns:
            int: Number of matched pairs.
        """
        return sum(len(pairs) for pairs in self.find_pairs().values())

    def class_count(self) -> Dict[str, int]:
        """
        Returns a dictionary with the count of pairs for each class.
        The keys are class names and the values are the counts.

        Returns:
            Dict[str, int]: Dictionary with class names as keys and
            counts as values.
        """
        return {cls: len(pairs) for cls, pairs in self.find_pairs().items()}

    def _load_pairs(self) -> None:
        """
        Private method to scan directories of given classes and match all
        available image-label pairs.
        It ignores images with missing labels.

        Returns:
            None
        """
        total_count = 0
        for cls in self._classes:
            img_dir = self._images_dir / cls

            if not Path.is_dir(img_dir):
                logger.info(f"Warning: Missing folder for {cls}. Skipping.")
                continue

            self._pairs[cls] = list(img_dir.glob("*.png"))
            N = len(self._pairs[cls])
            logger.info(f"Found {N} images in {cls}.")
            total_count += N

        print(f"Total pairs loaded: {total_count}")
