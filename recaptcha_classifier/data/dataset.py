class DatasetLoader:
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir

    def load_data(self):
        # Load image-label pairs
        pass

    def verify_annotations(self):
        # check YOLO annotations
        pass

    def combine_classes(self, target_classes):
        # combine dataset classes
        pass

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        # split dataset into train, val, test sets
        pass

    def balance_classes(self):
        # oversample underrepresented classes
        pass
