from pathlib import Path
from typing import List, Tuple


class DatasetLoader:
    def __init__(self, data_dir: str, label_dir: str):
        self.data: List[Tuple[str, str]] = []  # (img_path, label_path)
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.classes = ["Chimney", "Crosswalk", "Stair"]

    def load_data(self) -> List[Tuple[Path, Path]]:
        # Load image-label pairs
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
                    self.data.append((img_path, lbl_path))
                else:
                    missing += 1

            if missing > 0:
                print(f"Warning: {missing} missing labels in {cls}. Skipping.")

        print(f"Loaded {len(self.data)} image-label pairs.")
        return self.data

    def verify_annotations(self):
        # check YOLO annotations
        pass

    def combine_classes(self, target_classes):
        # combine dataset classes
        pass

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        # split dataset into train, val, test sets
        pass

    def balance_classes(self):
        # oversample underrepresented classes
        pass


if __name__ == "__main__":
    test = DatasetLoader("../../data/raw", "../../data/labels")
    print(test.load_data()[:10])
