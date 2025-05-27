import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from recaptcha_classifier.models.simple_classifier_model import SimpleCNN
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.data.pipeline import DataPreprocessingPipeline
from recaptcha_classifier.train.training import Trainer


class SimpleClassifierPipeline:
    def __init__(self,
                 step_size: int = 5,
                 gamma: float = 0.5,
                 lr: float = 0.001,
                 epochs: int = 15,
                 device: torch.device | None = None,
                 save_folder: str = "",
                 model_file_name: str = "model.pt",
                 optimizer_file_name: str = "optimizer.pt",
                 scheduler_file_name: str = "scheduler.pt"
                 ):

        self._class_map = DetectionLabels.to_class_map()
        self._data = DataPreprocessingPipeline(self.class_map, balance=True)
        self._loaders = self.data.run()
        print("Data loaders built successfully.")
        
        self._model = SimpleCNN(num_classes=len(self.class_map))
        self.optimizer = optim.RAdam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        self._trainer = Trainer(train_loader=self.loaders['train'],
                           val_loader=self.loaders['val'],
                           epochs=epochs,
                           optimizer=self.optimizer,
                           scheduler=self.scheduler,
                           save_folder=save_folder,
                           model_file_name=model_file_name,
                           optimizer_file_name=optimizer_file_name,
                           scheduler_file_name=scheduler_file_name,
                           device=device)

