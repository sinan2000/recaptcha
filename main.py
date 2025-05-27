from recaptcha_classifier import (
    DetectionLabels,
    DataPreprocessingPipeline,
    SimpleClassifierPipeline
)

def main():
    pipeline = SimpleClassifierPipeline(epochs=1)
    pipeline.train()
    pipeline.evaluate()


    # data = DataPreprocessingPipeline(
    #     DetectionLabels.to_class_map(),
    #     balance=True
    #     )

    # loaders = data.run()

    # print("Data loaders built successfully.")

    # for split, loader in loaders.items():
    #     print(f"{split.upper()} DataLoader:")
    #     batch = next(iter(loader))
    #     images, labels = batch
    #     print(f" - images.shape: {images.shape}")
    #     print(f" - labels.shape: {labels.shape}")
    #     print(f" - class IDs: {labels.tolist()}")



if __name__ == '__main__':
    main()
