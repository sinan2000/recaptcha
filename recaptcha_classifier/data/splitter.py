import random
from typing import Tuple
from .types import ClassFileDict, DatasetSplitDict, FilePairList


class DataSplitter:
    """
    Shuffles and splits data into train/validation/test sets.
    It respects the ratios provided by the user and distributions of the
    classes, so that the splits are stratified.

    Follows Single Responsibility Principle, only focusing
    on handling the data splliting logic.
    """
    def __init__(self,
                 ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 shuffle: bool = True,
                 seed: int = None) -> None:
        """
        Args:
            ratios (Tuple[float, float, float]): Ratios for
            train, validation, and test sets.
            shuffle (bool): Whether to shuffle the data before splitting.
            seed (int): Random seed for reproducibility.
        """
        self._ratios = ratios
        self._shuffle = shuffle
        self._seed = seed
        self._validate_ratios()

    def split(self,
              pairs_by_class: ClassFileDict
              ) -> DatasetSplitDict:
        """
        Splits each class into train, validation, and test sets.
        It shuffles the data if specified and returns a dictionary
        containing the splits by class.

        Args:
            items (List): List of items to be split.

        Returns:
            DatasetSplitDict: Nested dictionary containing
            splits for each class.
        """
        splits = {'train': {}, 'val': {}, 'test': {}}

        for cls, pairs in pairs_by_class.items():
            if self._shuffle:
                pairs = self._shuffle_items(pairs)

            N = len(pairs)
            train_N, val_N = self._get_split_sizes(N)

            splits['train'][cls] = pairs[:train_N]
            splits['val'][cls] = pairs[train_N:train_N + val_N]
            splits['test'][cls] = pairs[train_N + val_N:]

        return splits

    def _validate_ratios(self) -> None:
        """
        Makes sure the ratios sum to 1 and all are positive.
        """
        if len(self._ratios) != 3:
            raise ValueError("Ratios must be a tuple of three floats.")
        if sum(self._ratios) != 1:
            raise ValueError("Ratios must sum to 1.")
        if any(ratio < 0 for ratio in self._ratios):
            raise ValueError("Ratios must be positive.")

    def _shuffle_items(self, items: FilePairList) -> FilePairList:
        """
        Returns a shuffled copied version of the items list,
        using seed if provided.

        Args:
            items (FilePairList): List of items to be shuffled.

        Returns:
            FilePairList: Shuffled list of items.
        """
        new_items = items.copy()
        rand = random.Random(self._seed)
        rand.shuffle(new_items)
        return new_items

    def _get_split_sizes(self, n: int) -> Tuple[int, int]:
        """
        Returns the sizes of the train and validation sets.

        Args:
            n (int): Total number of items.

        Returns:
            Tuple[int, int]: Sizes of train and validation sets.
        """
        train_size = int(n * self._ratios[0])
        val_size = int(n * self._ratios[1])
        return train_size, val_size
