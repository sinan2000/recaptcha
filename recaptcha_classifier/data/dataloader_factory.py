from typing import Dict, Optional
from torch.utils.data import DataLoader
from .dataset import DatasetHandler
from .preprocessor import Preprocessor
from .augment import AugmentationPipeline
from .types import DatasetSplitDict, FilePairList
from .collate_batch import collate_batch


class DataLoaderFactory:
    """
    The class responsible for creating the DataLoader objects that will be
    passed to the training and validation loops.
    """
    def __init__(self,
                 class_map: dict,
                 preprocessor: Preprocessor,
                 augmentator: Optional[AugmentationPipeline] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 balance: bool = False) -> None:
        """
        Initializes the DataLoaderFactory with the given parameters.

        Args:
            class_map (dict): A dictionary mapping class names to indices.
            preprocessor (Preprocessor): The preprocessor to use.
            augmentator (Optional[AugmentationPipeline]): The augmentator used
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            balance (bool): Whether to balance the dataset.
        """
        self._preprocessor = preprocessor
        self._aug = augmentator
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._balance = balance
        self._class_map = class_map

    def create_loaders(self,
                       splits: DatasetSplitDict) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}

        for split_name, cls_dict in splits.items():
            # flatten nested dict of pairs
            flat_pairs: FilePairList = [pair
                                        # traversing over classes
                                        for pairs in cls_dict.values()
                                        # traversing over pairs
                                        for pair in pairs
                                        ]

            # augmentatr only for training set
            augmentator = self._aug if split_name == 'train' else None

            dataset = DatasetHandler(
                pairs=flat_pairs,
                preprocessor=self._preprocessor,
                augmentator=augmentator,
                class_map=self._class_map
            )

            loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=(split_name == 'train'),
                num_workers=self._num_workers,
                collate_fn=collate_batch
            )

            loaders[split_name] = loader

        return loaders
