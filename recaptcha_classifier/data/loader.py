import time
from pathlib import Path
from typing import List, Tuple, Iterator


class PairsLoader:
    """
    This class loads all image-label pairs for given classes.

    It follows Single Responsibility Principle (SRP) as it only handles
    the loading of the pairs. Also, it uses the Iterator pattern, as
    it can be looped over to get the pairs as tuples (img_path, lbl_path).
    """
    def __init__(self,
                 classes: List[str],
                 images_dir: str = "../../data/images",
                 labels_dir: str = "../../data/labels") -> None:
        """
        Initializes the PairsLoader instance.

        Args:
            images_dir (str): Path to the directory containing images.
            labels_dir (str): Path to the directory containing labels.
            classes (List[str]): List of class names to load.
        """
        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self._classes = classes
        self._pairs: List[Tuple[Path, Path]] = []

    def find_pairs(self) -> List[Tuple[Path, Path]]:
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

    def __iter__(self) -> Iterator[Tuple[Path, Path]]:
        """Iterates over (img_path, lbl_path) pairs."""
        return iter(self.find_pairs())

    def __len__(self) -> int:
        """
        Returns total number of matched pairs.

        Returns:
            int: Number of matched pairs.
        """
        return len(self.find_pairs())

    def _load_pairs(self) -> None:
        """
        Private method to scan directories of given classes and match all
        available image-label pairs.
        It ignores images with missing labels.
        """
        for cls in self._classes:
            img_dir = self._images_dir / cls
            lbl_dir = self._labels_dir / cls
            skipped, total = 0, 0

            if not img_dir.is_dir() or not lbl_dir.exists():
                print(f"Warning: Missing folder for {cls}. Skipping.")
                continue

            for img_path in img_dir.glob("*.png"):
                lbl_path = lbl_dir / img_path.name.replace(".png", ".txt")
                total += 1

                if not lbl_path.exists():
                    skipped += 1
                    continue

                self._pairs.append((img_path, lbl_path))

            print(f"Loaded {total - skipped} image-label pairs for {cls}.")
            if skipped > 0:
                print(f"Warning: {skipped} missing labels in {cls}. Skipped.")
            time.sleep(0.2)  # again, nice to have

        print(f"Total pairs loaded: {len(self._pairs)}")


if __name__ == "__main__":
    classes = ["Chimney", "Crosswalk", "Stair"]

    loader = PairsLoader(classes)
    pairs = loader.find_pairs()

    #  for img_path, lbl_path in pairs:
    #      print(f"Image: {img_path}, Label: {lbl_path}")
