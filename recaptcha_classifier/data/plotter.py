import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path


class SplitPlotter:
    """
    Helper classes that handles the plotting of the dataset splits.
    It is responsible for auditing the dataset splits, by ensuring
    visualization of classes across the splits.
    """
    def __init__(self,
                 splits: Dict[str, Dict[str, List[Tuple[Path, Path]]]]
                 ) -> None:
        self._splits = splits

    def print_counts(self) -> None:
        """
        Prints the counts of samples in each class, for each split.
        """
        for split, cls_dict in self._splits.items():
            print(f"{split.upper()}:")
            for cls, pairs in cls_dict.items():
                print(f"  {cls:5s}: {len(pairs)}")
            print('\n')

    def plot_splits(self,
                    title: str = "Class Distribution in Dataset Splits"
                    ) -> None:
        """
        Plots a bar chart showing the amount and percentage of samples
        present in each class, for each of the splits.
        """
        classes = list(self._splits['train'].keys())

        x = range(len(classes))
        width = 0.3

        # list of counts for each class
        counts_train = [len(self._splits['train'][cls]) for cls in classes]
        counts_val = [len(self._splits['val'][cls]) for cls in classes]
        counts_test = [len(self._splits['test'][cls]) for cls in classes]

        # total for each class, simply adding counts of each split
        totals = [t+v+te for t, v, te in
                  zip(counts_train, counts_val, counts_test)]

        # drawing bars
        train_bars = plt.bar([i - width for i in x],
                             counts_train,
                             width,
                             label='Train')
        val_bars = plt.bar(x, counts_val, width, label='Val')
        test_bars = plt.bar([i + width for i in x],
                            counts_test,
                            width,
                            label='Test')

        # adding perc labels on top of each of the bar
        for bars, counts in [(train_bars, counts_train),
                             (val_bars, counts_val),
                             (test_bars, counts_test)]:
            for bar, count, total in zip(bars, counts, totals):
                perc = count / total * 100
                plt.text(
                    bar.get_x() + bar.get_width() / 2,  # middle of the bar,
                    bar.get_height(),  # on top of the bar,
                    f'{perc:.1f}%',  # percentage, formatted to 1 decimal
                    ha='center', va='bottom'
                )

        plt.xticks(x, classes)
        plt.ylabel('No. of Samples')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
