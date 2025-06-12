from pathlib import Path
from typing import List, Dict
from .types import ClassToImgPaths


class ImagePathsLoader:
    """
    This class loads all image paths for given classes.

    It scans for all matching images and caches the result.

    It follows Single Responsibility Principle (SRP) as it only handles
    the loading of the image paths. Also, it uses the Iterator pattern, as
    it can be looped over to get the list paths as tuples by class,
    in format (class, [img_path1, ...]).
    """
    def __init__(self,
                 classes: List[str],
                 images_dir: str = "data") -> None:
        """
        Initializes the ImagePathsLoader instance.

        Args:
            classes (List[str]): List of class names to load.
            images_dir (str): Path to the directory containing images.
        """
        self._classes = classes
        self._images_dir = Path(images_dir)
        self._paths: ClassToImgPaths = dict()

    def find_image_paths(self) -> ClassToImgPaths:
        """
        Returns all image paths loaded for the given classes.
        It caches the response after first run.

        Returns:
            ClassToImgPaths: Dictionary mapping class names to lists
            of image paths.
        """
        if not self._paths:
            self._load_pairs()
        return self._paths

    def __iter__(self):
        """
        Iterates over classes and their respective list of pairs.

        Yields:
            Tuple[str, List[Path]]: Class name and list of
            image paths.
        """
        for cls, pairs in self.find_image_paths().items():
            yield cls, pairs

    def __len__(self) -> int:
        """
        Returns total number of matched pairs.

        Returns:
            int: Number of matched pairs.
        """
        return sum(len(pairs) for pairs in self.find_image_paths().values())

    def class_count(self) -> Dict[str, int]:
        """
        Returns a dictionary with the count of pairs for each class.
        The keys are class names and the values are the counts.

        Returns:
            Dict[str, int]: Dictionary with class names as keys and
            counts as values.
        """
        return {cls: len(
            pairs) for cls, pairs in self.find_image_paths().items()}

    def _load_pairs(self) -> None:
        """
        Private method to scan directories of given classes and match all
        available image paths.
        """
        total_count = 0
        for cls in self._classes:
            img_dir = self._images_dir / cls

            if not Path.is_dir(img_dir):
                print(f"Warning: Missing folder for {cls}. Skipping.")
                continue

            self._paths[cls] = sorted(img_dir.glob("*.png"))
            N = len(self._paths[cls])
            print(f"Found {N} images in {cls}.")
            total_count += N

        print(f"Total image paths found: {total_count}")
