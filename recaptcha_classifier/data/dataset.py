from pathlib import Path
from typing import List, Tuple, Dict


class DatasetLoader:
    """
    This class simplifies data loading, validation and combination
    into a single interface. It hides the complexity of loading
    images and labels from the user.

    Design Pattern: Facade
    """

    def __init__(self,
                 data_dir: str,
                 label_dir: str,
                 ) -> None:
        """
        Initialize the DatasetLoader.

        Args:
            data_dir (str): Path to raw image directory.
            label_dir (str): Path to label directory.
        """
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.classes = ["Chimney", "Crosswalk", "Stair"]
        self.data: Dict[str, List[Tuple[str, str]]] = {cls: [] for
                                                       cls in self.classes}
        self.expected_size = (120, 120)

    def load_data(self) -> None:
        """
        Load available image-label pairs for the given classes. Ignores
        images with missing labels.
        """
        for cls in self.classes:
            self._load_class_data(cls)

    def _load_class_data(self, cls: str) -> None:
        """
        Load images and labels for a specific class. Ignores images
        with missing labels.

        Args:
            cls (str): Class name.
        """
        img_dir = self.data_dir / cls
        lbl_dir = self.label_dir / cls

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Warning: Missing folder for {cls}. Skipping.")
            return

        missing = 0
        for img_path in img_dir.glob("*.png"):
            lbl_path = lbl_dir / img_path.name.replace(".png", ".txt")

            if not lbl_path.exists():
                missing += 1
                continue

            self.data[cls].append((img_path, lbl_path))

        print(f"Loaded {len(self.data[cls])} image-label pairs for {cls}.")
        if missing > 0:
            print(f"Warning: {missing} missing labels in {cls}. Skipped.")

    def verify_annotations(self) -> None:
        """
        Verify the annotations of the loaded YOLO files.
        """
        for cls, pairs in self.data.items():
            for _, lbl_path in pairs:
                with open(lbl_path, "r") as file:
                    for line in file.readlines():
                        b = [float(x) for x in line.strip().split()]
                        if len(b) != 5:
                            print("Warning: " +
                                  f"Invalid bounding box in {lbl_path}.")
                            break
                        for i in range(1, 5):
                            if b[i] < 0 or b[i] > 1:
                                print("Warning: " +
                                      f"Invalid bounding box in {lbl_path}.")
                                break

    def combine_classes(self,
                        target_classes: List[str]
                        ) -> List[Tuple[Path, Path]]:
        """
        Combine data from multiple classes.

        Args:
            target_classes (List[str]): List of classes to combine.

        Returns:
            List[Tuple[Path, Path]]: Combined image-label pairs.
        """
        pairs = []
        for cls in target_classes:
            if cls not in self.data:
                print(f"Warning: Class {cls} not found. Skipping.")
                continue

            pairs.extend(self.data[cls])

        print(f"Combined {len(pairs)} image-label " +
              f"pairs from classes: {target_classes}.")

        return pairs

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
    test.load_data()
    comb = test.combine_classes(["Chimney", "Crosswalk"])
    test.verify_annotations()
    print(comb[:10])
