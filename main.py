# from recaptcha_classifier import (
#    DetectionLabels,
#    DataPreprocessingPipeline
# )


def hello_world():
    return "Hello, World!"


if __name__ == '__main__':
    hello_world()
    """
    wrapper = DataPreprocessingPipeline(DetectionLabels.to_class_map())
    loaders = wrapper.run()
    print("Data loaders created successfully.")

    for split, loader in loaders.items():
        print(f"{split.upper()} DataLoader:")
        for batch in loader:
            images, bboxes, labels = batch
            print(f"First image shape: {images[0].shape}")
            print(f"First image dtype: {images[0].dtype}")

            print(f"First bounding boxes: {bboxes[0]}")

            print(f"First label (class index): {labels[0]}")

            # Check batch sizes
            print(f"Batch size: {len(images)}")
            print(f"Total images in batch: {len(images)}")

            print("-" * 40)
            break
    """
