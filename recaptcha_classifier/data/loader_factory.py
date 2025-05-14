from typing import Dict, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from .dataset import ImageDataset
from .preprocessor import ImagePrep
from .augment import AugmentationPipeline
from .types import DatasetSplitDict, FilePairList
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
            preprocessor (ImagePrep): The preprocessor to use.
            augmentator (Optional[AugmentationPipeline]): The augmentator used
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            balance (bool): Whether to balance the dataset.
        """
        self._preprocessor = preprocessor
        self._aug = augmentator
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._balance = balance  # do we use WeightedRandomSampler??
        self._class_map = class_map
        #  OPTIONAL: self._loaders to cache response

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

            dataset = ImageDataset(
                pairs=flat_pairs,
                preprocessor=self._preprocessor,
                augmentator=augmentator,
                class_map=self._class_map
            )

            sampler = (self._build_sampler(flat_pairs) if self._balance
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

    def _build_sampler(self, pairs):
        """
        Builds a sampler for the dataset to balance the classes.
        """
        """
        class_counts = Counter([class_map[pair[1].parent.name]
                                for pair in pairs])
        weights = [1.0 / class_counts[class_map[pair[1].parent.name]]
                   for pair in pairs]
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler
        """
        class_counts = Counter()
        targets = []
        for _, lbl_path in pairs:
            cls = lbl_path.parent.name
            targets.append(self._class_map[cls])
            class_counts[self._class_map[cls]] += 1

        total = sum(class_counts.values())
        weights = {k: total / v for k, v in class_counts.items()}
        sample_weights = [weights[target] for target in targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler
