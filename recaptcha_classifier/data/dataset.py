from pathlib import Path
from typing import List, Tuple
from PIL import Image


class DatasetLoader:
    """
    Facade Pattern:
    This class simplifies data loading, validation and combination
    into a single interface. It hides the complexity of loading
    images and labels from the user.
    """

    def __init__(self, data_dir: str, label_dir: str) -> None:
        """
        Initialize the DatasetLoader.

        Args:
            data_dir (str): Path to raw image directory.
            label_dir (str): Path to label directory.
        """
        self.data: List[Tuple[str, str]] = []  # (img_path, label_path)
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.classes = ["Chimney", "Crosswalk", "Stair"]
        self.expected_size = (120, 120)
        self.wrong_size_imgs = 0

    def load_data(self) -> List[Tuple[Path, Path]]:
        """
        Load available image-label pairs for the given classes. Ignores
        images with missing labels.

        Returns:
            List[Tuple[Path, Path]]: List of tuples containing image and
            label paths.
        """
        for cls in self.classes:
            missing = 0
            img_dir = self.data_dir / cls
            lbl_dir = self.label_dir / cls

            if not img_dir.exists() or not lbl_dir.exists():
                print(f"Warning: Missing folder for {cls}. Skipping.")
                continue

            for img_path in img_dir.glob("*.png"):
                lbl_path = lbl_dir / img_path.name.replace(".png", ".txt")

                if lbl_path.exists():
                    with Image.open(img_path) as img:
                        if img.size != self.expected_size:
                            self.wrong_size_imgs += 1
                            continue
                    self.data.append((img_path, lbl_path))
                else:
                    missing += 1

            if missing > 0:
                print(f"Warning: {missing} missing labels in {cls}. Skipping.")

        print(f"Loaded {len(self.data)} image-label pairs.")
        print(f"Total images with wrong size: {self.wrong_size_imgs}")
        return self.data

    def verify_annotations(self):
        """
        Verify the annotations of the loaded YOLO files.
        """
        pass

    def combine_classes(self, target_classes):
        """
        Combine data from multiple classes.

        Args:
            target_classes (List[str]): List of classes to combine.
        """
        # combine dataset classes
        pass

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Split into train, val and test sets.

        Args:
            train_ratio (float): Ratio for training set.
            val_ratio (float): Ratio for validation set.
            test_ratio (float): Ratio for test set.
        """
        pass

    def balance_classes(self):
        """
        Oversample underrepresented classes to prevent class imbalance.
        """
        pass


if __name__ == "__main__":
    test = DatasetLoader("../../data/raw", "../../data/labels")
    print(test.load_data()[:10])
