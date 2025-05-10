from typing import Dict, Optional, Any  # List, Tuple,
from torch.utils.data import DataLoader
# import torch
# from .dataset import DatasetHandler
from .preprocessor import Preprocessor
from .augment import AugmentationPipeline


class DataLoaderFactory:
    """
    The class responsible for creating the DataLoader objects that will be
    passed to the training and validation loops.

    It uses the Factory design pattern, because...
    """
    def __init__(self,
                 preprocessor: Preprocessor,
                 augmentator: Optional[AugmentationPipeline] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 balance: bool = False) -> None:
        """
        Initializes the DataLoaderFactory with the given parameters.

        Args:
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

    def create_loaders(self,
                       splits: Dict[str, Any]) -> Dict[str, DataLoader]:
        # loaders: Dict[str, DataLoader] = {}
        pass
        # for split, pairs in splits.items():
        # we will implement here the loader creation
        # however, we need to check all interfaces arguments to match
        # in order to integrate the whole pipeline first
