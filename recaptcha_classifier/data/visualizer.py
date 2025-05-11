import matplotlib.pyplot as plt
from .types import DatasetSplitDict


class Visualizer:
    """
    Helper class that handles the plotting of the dataset splits.
    It is responsible for auditing the dataset splits, by ensuring
    visualization of classes across the splits.
    """
    @classmethod
    def print_counts(cls, splits: DatasetSplitDict) -> None:
        """
        Prints the counts of samples in each class, for each split.

        Args:
            splits (DatasetSplitDict): the dataset splits
                containing the pairs for each class.
        """
        for split, cls_dict in splits.items():
            print(f"{split.upper()}:")
            for cls, pairs in cls_dict.items():
                print(f"  {cls:5s}: {len(pairs)}")
            print('\n')

    @classmethod
    def plot_splits(cls,
                    splits: DatasetSplitDict,
                    title: str = "Class Distribution in Splits") -> None:
        """
        Plots a bar chart showing the amount and percentage of samples
        present in each class, for each of the splits.

        Args:
            splits (DatasetSplitDict): the dataset splits
                containing the pairs for each class.
            title (str): the title of the plot.
        """
        classes = list(splits['train'].keys())

        x = range(len(classes))
        width = 0.3

        # list of counts for each class
        counts_train = [len(splits['train'][cls]) for cls in classes]
        counts_val = [len(splits['val'][cls]) for cls in classes]
        counts_test = [len(splits['test'][cls]) for cls in classes]

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
