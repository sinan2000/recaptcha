from typing import Dict, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from .dataset import ImageDataset
from .preprocessor import ImagePrep
from .augment import AugmentationPipeline
from .types import DatasetSplitMap, ImagePathList
from .collate_batch import collate_batch


class LoaderFactory:
    """
    The class responsible for creating the DataLoader objects for each split
    of the dataset.
    """
    def __init__(self,
                 class_map: dict,
                 preprocessor: ImagePrep,
                 augmentator: Optional[AugmentationPipeline] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 balance: bool = False) -> None:
        """
        Initializes the LoaderFactory with the given parameters.

        Args:
            class_map (dict): A dictionary mapping class names to indices.
            preprocessor (ImagePrep): The preprocessor to use for data.
            augmentator (Optional[AugmentationPipeline]): The augmentator used
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            balance (bool): Whether to balance the dataset.

        Returns:
            None
        """
        self._preprocessor = preprocessor
        self._aug = augmentator
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._balance = balance
        self._class_map = class_map

    def create_loaders(self,
                       splits: DatasetSplitMap) -> Dict[str, DataLoader]:
        """
        Creates the DataLoader objects for each split of the dataset.

        Args:
            splits (DatasetSplitDict): A dictionary containing the train,
            val, and test splits of the dataset.

        Returns:
            Dict[str, DataLoader]: A dictionary containing the DataLoader
            objects for each split of the dataset.
        """
        loaders: Dict[str, DataLoader] = {}

        for split_name, cls_dict in splits.items():
            # flatten nested dict of image_paths
            flat_image_paths: ImagePathList = [
                image_path
                # traversing over classes
                for image_paths in cls_dict.values()
                # traversing over image_paths
                for image_path in image_paths
                ]

            # augmentatr only for training set
            augmentator = self._aug if split_name == 'train' else None

            dataset = ImageDataset(
                items=flat_image_paths,
                preprocessor=self._preprocessor,
                augmentator=augmentator,
                class_map=self._class_map
            )

            sampler = (self._build_sampler(flat_image_paths) if self._balance
                       and split_name == "train" else None)

            loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=(split_name == 'train' and not sampler),
                num_workers=self._num_workers,
                collate_fn=collate_batch,
                sampler=sampler
            )

            loaders[split_name] = loader

        return loaders

    def _build_sampler(
         self, image_paths: ImagePathList) -> WeightedRandomSampler:
        """
        Builds a sampler for the dataset to balance the classes.

        Args:
            image_paths (ImagePathList): A list of image paths.

        Returns:
            WeightedRandomSampler: A sampler for the dataset.
        """
        class_counts = Counter()
        targets = []
        for img_path in image_paths:
            cls = img_path.parent.name
            targets.append(self._class_map[cls])
            class_counts[self._class_map[cls]] += 1

        total = sum(class_counts.values())
        weights = {k: total / v for k, v in class_counts.items()}
        sample_weights = [weights[target] for target in targets]
        norm = sum(sample_weights)
        sample_weights = [w / norm for w in sample_weights]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler
