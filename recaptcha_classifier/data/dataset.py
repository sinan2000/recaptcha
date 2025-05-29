from typing import Optional
from torch.utils.data import Dataset
from .preprocessor import ImagePrep
from .augment import AugmentationPipeline
from .types import FilePairList, DataItem


class ImageDataset(Dataset):
    """
    A class to handle the pairs of (image, label) for the dataset.
    It makes them ready for training, by applying augmentation
    for the training set, preprocessing and makes sure that the
    output format is in PyTorch Tensor format.

    It has the required methods to be used as a inherited class
    of Dataset, as per the PyTorch documentation: (for custom datasets)
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self,
                 pairs: FilePairList,
                 preprocessor: ImagePrep,
                 augmentator: Optional[AugmentationPipeline] = None,
                 class_map: dict = {}
                 ) -> None:
        """
        Initializes the ImageDataset with the given parameters.

        Args:
            pairs (FilePairList): A list of tuples containing the
            image and label file paths.
            preprocessor (ImagePrep): An instance of the
            ImagePrep class for preprocessing the images.
            augmentator (Optional[AugmentationPipeline]): An
            instance of the AugmentationPipeline class for applying
            augmentations to the images.
            class_map (dict): A dictionary mapping class names to indices.

        Returns:
            None
        """
        self._pairs = pairs
        self._prep = preprocessor
        self._aug = augmentator
        self._class_map = class_map

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._pairs)

    def __getitem__(self,
                    idx: int
                    ) -> DataItem:
        """
        Returns the item at the given index, ready for training.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            DataItem: A tuple containing the
            preprocessed image in tensor format, the YOLO bound
            box annotations and the label.
        """
        # img_path, lbl_path = self._pairs[idx]
        img_path = self._pairs[idx]     

        # Load image and label
        img = self._prep.load_image(img_path)
        # bb = self._prep.load_labels(lbl_path)

        # if not bb:
        #    raise ValueError(f"Bounding box list is empty for {lbl_path}")

        # Apply augmentation if passed
        if self._aug:
            # img, bb = self._aug.apply_transforms(img, bb)
            img, _ = self._aug.apply_transforms(img, [])

        # Convert image to tensor
        tensor = self._prep.to_tensor(img)

        # Convert label to class index
        c_name = img_path.parent.name
        if c_name not in self._class_map:
            raise KeyError(f"Class name '{c_name}' not found in classes.")
        c_id = self._class_map[c_name]

        # Return image tensor, bounding box and class index
        return tensor, self._prep.class_id_to_tensor(c_id)
        # return tensor, bb, c_id
