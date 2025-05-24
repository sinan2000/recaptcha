import matplotlib.pyplot as plt
from .types import DatasetSplitMap
import numpy as np


class Visualizer:
    """
    Helper class that handles the plotting of the dataset splits.
    It is responsible for auditing the dataset splits, by ensuring
    visualization of classes across the splits.
    """
    @classmethod
    def print_counts(cls, splits: DatasetSplitMap) -> None:
        """
        Prints the counts of samples in each class, for each split.

        Args:
            splits (DatasetSplitMap): the dataset splits
                containing the items for each class.
        """
        for split, cls_dict in splits.items():
            print(f"{split.upper()}:")
            for cls, items in cls_dict.items():
                print(f"  {cls:5s}: {len(items)}")
            print()

    @classmethod
    def plot_splits(cls,
                    splits: DatasetSplitMap,
                    title: str = "Class Distribution in Splits") -> None:
        """
        Plots a bar chart showing the amount and percentage of samples
        present in each class, for each of the splits.

        Args:
            splits (DatasetSplitMap): the dataset splits
                containing the items for each class.
            title (str): the title of the plot.
        """
        classes = list(splits['train'].keys())

        num_classes = len(classes)
        x = np.arange(num_classes)  # range(len(classes))
        width = 0.3

        # list of counts for each class
        counts_train = [len(splits['train'][cls]) for cls in classes]
        counts_val = [len(splits['val'][cls]) for cls in classes]
        counts_test = [len(splits['test'][cls]) for cls in classes]

        # total for each class, simply adding counts of each split
        totals = [t+v+te for t, v, te in
                  zip(counts_train, counts_val, counts_test)]
        
        _, ax = plt.subplots(figsize=(num_classes, 6))
        bar1 = ax.bar(x - width, counts_train, width, label='Train')
        bar2 = ax.bar(x, counts_val, width, label='Val')
        bar3 = ax.bar(x + width, counts_test, width, label='Test')

        def annotate(bars, counts):
            for bar, count, total in zip(bars, counts, totals):
                p = 100 * count / total if total else 0
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height()+3,
                        f'{p:.1f}%',
                        ha='center', va='bottom', fontsize=8)

        annotate(bar1, counts_train)
        annotate(bar2, counts_val)
        annotate(bar3, counts_test)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('No. of Samples', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
