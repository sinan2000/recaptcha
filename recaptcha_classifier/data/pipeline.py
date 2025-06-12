from typing import Dict, Tuple
from torch.utils.data import DataLoader
from enum import EnumMeta
from .downloader import DatasetDownloader
from .paths_loader import ImagePathsLoader
from .splitter import DataSplitter
from .visualizer import Visualizer
from .preprocessor import ImagePrep
from .augment import AugmentationPipeline
from torchvision import transforms
from .loader_factory import LoaderFactory


class DataPreprocessingPipeline:
    """
    High-level wrapper, implemented as an interface for the entire
    data preprocessing pipeline.
    It integrates all components in order to ensure dataset is
    downloaded, split and prepared into PyTorch DataLoaders for training.

    It handles:
    1. Downloading the dataset if not already present locally.
    2. Finds all pairs of images and YOLO annotations from the dataset
    3. Splitting dataset into train/val/test
    4. Plotting dataset distribution (optionally)
    5. Creating the Pytorch format DataLoaders for each split

    It is implemented using the Facade Pattern, as it is a simple
    interface that handles all the complexity of the data preprocessing
    pipeline and it includes all its components.
    """
    def __init__(self,
                 class_enum: EnumMeta,
                 ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 seed: int = 23,  # our group number
                 batch_size: int = 32,
                 num_workers: int = 4,
                 balance: bool = True,
                 show_plots: bool = False) -> None:
        """
        Initializes the DataPreprocessingPipeline with the given parameters.

        Args:
            class_enum (EnumMeta): Enum class containing dataset classes.
            ratios (Tuple[float, float, float]): Ratios for train,
            val, and test splits.
            seed (int): Random seed for reproducibility.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            balance (bool): Whether to balance the dataset.
            show_plots (bool): Whether to show plots.

        Returns:
            None
        """
        self._class_enum = class_enum
        self._downloader = DatasetDownloader(self._class_enum
                                             .dataset_classnames())
        self._loader = ImagePathsLoader(self._class_enum.dataset_classnames())
        self._splitter = DataSplitter(ratios, seed=seed)
        self._show_plots = show_plots
        self._preproc = ImagePrep()
        self._augment = self._build_augmentator()
        self._creator = LoaderFactory(
            class_map=self._class_enum.to_class_map(),
            preprocessor=self._preproc,
            augmentator=self._augment,
            batch_size=batch_size,
            num_workers=num_workers,
            balance=balance
        )

    def _build_augmentator(self) -> AugmentationPipeline:
        """
        Builds the augmentation pipeline.
        One of these augmentations will be applied randomly
        in the dataset, for any image in the training set.

        Returns:
            AugmentationPipeline: The augmentation pipeline.
        """
        return AugmentationPipeline([
            # 50 % chance to flip the image horizontally
            transforms.RandomHorizontalFlip(p=0.5),
            # rotates image randomly within +-15 degrees
            transforms.RandomRotation(degrees=15),
            # randomly changes brightness, contrast and saturation
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3),
            # Translates/ shifts image up to 10% of its size
            # in both x and y directions
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # Mimics a camera lens blur effect
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            # Crops image 90% then resizes it to 224x224 (zoom in effect)
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        ])

    def run(self) -> Dict[str, DataLoader]:
        """
        Runs the entire preprocessing pipeline.

        Returns:
            Dict[str, DataLoader]: Dictionary containing DataLoaders for
            train, val, and test sets.
        """
        print("Running data preprocessing pipeline...")
        # 1. Downloads the dataset if not already present locally
        print("a. Checking local files...")
        self._downloader.download()

        # 2. Finds all pairs of images and YOLO annotations from the dataset
        print("b. Searching for all the data...")
        pairs_by_class = self._loader.find_image_paths()

        # 3a. Splits the data into train, val, and test sets
        print("c. Splitting the data...")
        splits = self._splitter.split(pairs_by_class)

        # 3b. Plots the dataset distribution, only if show_plots is True
        if self._show_plots:
            print("d. show_plots is True, plotting the dataset "
                  "distribution...")
            Visualizer.print_counts(splits)
            Visualizer.plot_splits(splits)
        else:
            print("d. show_plots is False, not plotting"
                  " the dataset distribution...")

        # 4. Create DataLoaders for each split
        print("e. Creating DataLoaders for each split...")
        loaders = self._creator.create_loaders(splits)
        return loaders
