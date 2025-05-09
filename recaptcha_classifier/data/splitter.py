import random
from typing import List, Dict, Tuple


class DatasetSplitter:
    """
    Shuffles and splits data into train/validation/test sets.

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

    def split(self, items: List) -> Dict[str, List]:
        """
        Splits the given list of items into train, validation, and test sets.

        Args:
            items (List): List of items to be split.

        Returns:
            Dict[str, List]: Dictionary containing train, validation,
            and test sets.
        """
        if self._shuffle:
            items = self._shuffle_items(items)

        N = len(items)
        train_N, val_N, _ = self._get_split_sizes(N)

        train_set = items[:train_N]
        val_set = items[train_N:train_N + val_N]
        test_set = items[train_N + val_N:]

        return {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }

    def _validate_ratios(self) -> None:
        """
        Makes sure the ratios sum to 1 and all are positive.
        """
        if sum(self._ratios) != 1:
            raise ValueError("Ratios must sum to 1.")
        if any(ratio < 0 for ratio in self._ratios):
            raise ValueError("Ratios must be positive.")

    def _shuffle_items(self, items: List) -> List:
        """
        Returns a shuffled copied version of the items list,
        using seed if provided.

        Args:
            items (List): List of items to be shuffled.

        Returns:
            List: Shuffled list of items.
        """
        new_items = items.copy()
        rand = random.Random(self._seed)
        rand.shuffle(new_items)
        return new_items

    def _get_split_sizes(self, n: int) -> Tuple[int, int, int]:
        """
        Returns the sizes of the train, validation, and test sets.

        Args:
            n (int): Total number of items.

        Returns:
            Tuple[int, int, int]: Sizes of train, validation,
            and test sets.
        """
        train_size = int(n * self._ratios[0])
        val_size = int(n * self._ratios[1])
        test_size = n - train_size - val_size
        return train_size, val_size, test_size
