from recaptcha_classifier import (
    DetectionLabels,
    DataPreprocessingPipeline
)


def main():
    pipeline = DataPreprocessingPipeline(
        DetectionLabels,
        balance=True
        )

    loaders = pipeline.run()
    
    print("Data loaders built successfully.")

    for split, loader in loaders.items():
        print(f"{split.upper()} DataLoader:")
        batch = next(iter(loader))
        images, labels = batch
        print(f" - images.shape: {images.shape}")
        print(f" - labels.shape: {labels.shape}")
        print(f" - class IDs: {labels.tolist()}")


if __name__ == '__main__':
    main()
