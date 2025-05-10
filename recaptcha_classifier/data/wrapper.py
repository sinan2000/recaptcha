from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from .downloader import KaggleDatasetDownloader
from .loader import PairsLoader
from .splitter import DatasetSplitter
# from .plotter import SplitPlotter
from .preprocessor import Preprocessor
from .augment import (
    AugmentationPipeline,
    HorizontalFlip,
    RandomRotation
)


class PreprocessingWrapper:
    """
    Facade wrap responsible for all preprocessing steps.
    It handles:
    1. Downloading the dataset
    2. Loading pairs from disk (data/ folder)
    3a. splitting dataset into train/val/test
    3b. plotting dataset distribution
    4. Creating the Pytorch format DataLoaders for each split
    """
    def __init__(self,
                 classes: List[str] = ["Chimney", "Crosswalk", "Stair"],
                 ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 seed: int = 23,  # our group number
                 batch_size: int = 32,
                 num_workers: int = 4,
                 balance: bool = False,
                 show_plots: bool = False) -> None:
        """
        Initializes the PreprocessingWrapper with the given parameters.

        Args:
            classes (List[str]): List of class names.
            ratios (Tuple[float, float, float]): Ratios for train,
            val, and test splits.
            seed (int): Random seed for reproducibility.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            balance (bool): Whether to balance the dataset.
            show_plots (bool): Whether to show plots.
        """
        self._downloader = KaggleDatasetDownloader()
        self._loader = PairsLoader(classes)
        self._splitter = DatasetSplitter(ratios, seed=seed)
        #  self._plotter = SplitPlotter(we need to pass the splits)
        self._show_plots = show_plots
        self._preproc = Preprocessor()
        self._augment = self._build_augmentator()
        """
        self._creator = DataLoaderCreator(
            preprocessor=self._preproc,
            augmentator=self._augment,
            batch_size=batch_size,
            num_workers=num_workers,
            balance=balance
        )
        """

    def _build_augmentator(self) -> AugmentationPipeline:
        """
        Builds the augmentation pipeline.

        Returns:
            AugmentationPipeline: The augmentation pipeline.
        """
        aug = AugmentationPipeline()

        aug.add_transform(HorizontalFlip(p=0.5))
        aug.add_transform(RandomRotation(degrees=30))

        return aug

    def run(self) -> Dict[str, DataLoader]:
        """
        Runs the entire preprocessing pipeline.

        Returns:
            Dict[str, DataLoader]: Dictionary containing DataLoaders for
            train, val, and test sets.
        """
        # 1. Downloads the dataset if not already present locally
        self._downloader.download()

        # 2. Loads pairs from disk, in dictionary for each class
        # pairs_by_class = self._loader.find_pairs()

        # 3a. Splits the data into train, val, and test sets
        # splits = self._splitter.split(pairs_by_class)

        # 3b. Plots the dataset distribution, only if show_plots is True
        if self._show_plots:
            # plotter = self._plotter(splits)
            # plotter.print_counts()
            # plotter.plot()
            pass

        # 4. Create DataLoaders for each split
        loaders = ['train', 'empty']  # self._creator.create(splits)
        return loaders
