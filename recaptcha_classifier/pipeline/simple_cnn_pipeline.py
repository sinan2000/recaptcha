import torch
from recaptcha_classifier.models.simple_classifier_model import SimpleCNN
from recaptcha_classifier.pipeline.base_pipeline import BasePipeline
from recaptcha_classifier.train.training import Trainer
import os


class SimpleClassifierPipeline(BasePipeline):
    def __init__(self,
                 lr: float = 0.001,
                 epochs: int = 20,
                 device: torch.device | None = None,
                 save_folder: str = "simple_classifier_checkpoints",
                 model_file_name: str = "simple_model.pt",
                 optimizer_file_name: str = "optimizer.pt",
                 scheduler_file_name: str = "scheduler.pt"
                 ):
        super().__init__(lr, epochs, device,
                         save_folder, model_file_name,
                         optimizer_file_name, scheduler_file_name)

    def run(self, save_train_checkpoints: bool = True,
            load_train_checkpoints: bool = False) -> None:
        """
        Runs the whole pipeline. Default saving of the training checkpoints.
        :param save_train_checkpoints: boolean to save the training checkpoints during training.
        :param load_train_checkpoints: boolean to load the training checkpoints during training.
        """
        self._data_loader()
        self._model = self._initialize_model()
        self._trainer = self._initialize_trainer()
        print("Training:")
        self._trainer.train(model=self._model)
        self.evaluate(plot_cm=True)

    def _initialize_model(self) -> SimpleCNN:
        return SimpleCNN(num_classes=self.class_map_length)
    
    def save_model(self):
        torch.save(self._model.state_dict(), os.path.join(self.save_folder, self.model_file_name))