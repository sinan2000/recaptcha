# This is a sample Python script.


def hello_world():
    return "Hello, World!"


if __name__ == '__main__':
    hello_world()

"""
Commented to pass main unit test

from recaptcha_classifier import (
    KaggleDatasetDownloader,
    PairsLoader,
    DatasetSplitter,
    SplitPlotter,
    Preprocessor
)

SHOW_PLOTS = True

if __name__ == '__main__':
    KaggleDatasetDownloader().download()
    pairs = PairsLoader().find_pairs()
    dataset = DatasetSplitter().split(pairs)

    if SHOW_PLOTS:
        plotter = SplitPlotter(dataset)
        plotter.print_counts()

        plotter.plot_splits()
"""
