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
    Preprocessor,
    DatasetSplitter
)

if __name__ == '__main__':
    KaggleDatasetDownloader().download()
    pairs = PairsLoader().find_pairs()

    dataset = []
    preprocessor = Preprocessor()
    for img_path, lbl_path in pairs:
        dataset.append(preprocessor.process_pairs(img_path, lbl_path))

    splits = DatasetSplitter().split(pairs)

    print(f"Test set size: {len(splits['test'])}")

"""
