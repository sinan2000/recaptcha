import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from recaptcha_classifier.models.simple_classifier_model import SimpleCNN
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.data.pipeline import DataPreprocessingPipeline
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model


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
        self._loaders = self._data.run()
        print("Data loaders built successfully.")
        
        self._model = SimpleCNN(num_classes=len(self.class_map))
        self.optimizer = optim.RAdam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        self._trainer = Trainer(train_loader=self._loaders['train'],
                                val_loader=self._loaders['val'],
                                epochs=epochs,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                save_folder=save_folder,
                                model_file_name=model_file_name,
                                optimizer_file_name=optimizer_file_name,
                                scheduler_file_name=scheduler_file_name,
                                device=device)

        # ????????
        self._load_checkpoint = (
            os.path.exists(os.path.join(save_folder, model_file_name)) and
            os.path.exists(os.path.join(save_folder, optimizer_file_name)) and
            os.path.exists(os.path.join(save_folder, scheduler_file_name))
        )

    def train(self) -> None:
        self._trainer.train(self._model, self._load_checkpoint, save_checkpoint=True)

    def evaluate(self, plot_cm: bool = False) -> dict:
        eval_results = evaluate_model(
            model=self._model,
            test_loader=self._loaders['test'],
            device=self._trainer.device,
            num_classes=len(self._class_map),
            class_names=list(self._class_map.keys()),
            plot_cm=plot_cm
        )
        return eval_results
