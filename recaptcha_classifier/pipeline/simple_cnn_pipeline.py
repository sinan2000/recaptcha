import torch
from recaptcha_classifier.models.simple_classifier_model import SimpleCNN
from recaptcha_classifier.pipeline.base_pipeline import BasePipeline
import os
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.constants import (
    MODELS_FOLDER, SIMPLE_MODEL_FILE_NAME,
    OPTIMIZER_FILE_NAME, SCHEDULER_FILE_NAME
)


class SimpleClassifierPipeline(BasePipeline):
    """ Pipeline for training a simple CNN model. """
    def __init__(self,
                 lr: float = 0.001,
                 epochs: int = 20,
                 device: torch.device | None = None,
                 save_folder: str = MODELS_FOLDER,
                 model_file_name: str = SIMPLE_MODEL_FILE_NAME,
                 optimizer_file_name: str = OPTIMIZER_FILE_NAME,
                 scheduler_file_name: str = SCHEDULER_FILE_NAME,
                 early_stopping: bool = True,
                 ) -> None:
        """
        Constructor for SimpleClassifierPipeline class.

        Args:
            step_size (int, optional): Step size for the learning rate
                scheduler. Defaults to 5.
            gamma (float, optional): Gamma value for the learning rate
                scheduler. Defaults to 0.5.
            lr (float, optional): Learning rate for the optimizer.
                Defaults to 0.001.
            epochs (int, optional): Number of epochs for training.
                Defaults to 20.
            device (torch.device, optional): Device for training.
                Defaults to None.
            save_folder (str, optional): Folder for saving checkpoint files.
                Defaults to "".
            model_file_name (str, optional): Name of the model checkpoint
                file. Defaults to "model.pt".
            optimizer_file_name (str, optional): Name of the optimizer
                checkpoint file. Defaults to "optimizer.pt".
            scheduler_file_name (str, optional): Name of the scheduler
                checkpoint file. Defaults to "scheduler.pt".
        """
        super().__init__(lr, epochs, device,
                         save_folder, model_file_name,
                         optimizer_file_name, scheduler_file_name,
                         early_stopping)

    def run(self, save_train_checkpoints: bool = True,
            load_train_checkpoints: bool = False) -> None:
        """
        Runs the whole pipeline. Default saving of the training checkpoints.
        Args:
            save_train_checkpoints (bool, optional): Whether to save the
                training checkpoints. Defaults to True.
            load_train_checkpoints (bool, optional): Whether to load the
                training checkpoints. Defaults to False.
        """
        self._data_loader()
        self._model = self._initialize_model()
        self._trainer = self._initialize_trainer()
        print("Training:")
        self._trainer.train(model=self._model)
        self.save_model()
        self.evaluate(plot_cm=True)

    def _initialize_model(self) -> SimpleCNN:
        """Initialize the model.

        Returns:
            SimpleCNN: The initialized model.
        """
        return SimpleCNN(num_classes=self.class_map_length)

    def save_model(self):
        """Saves the model.

        Returns:
            None
        """
        os.makedirs(self.save_folder, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(
            self.save_folder, self.model_file_name))

    def _initialize_trainer(self) -> Trainer:
        """Initialize the trainer.

        Returns:
            Trainer: The initialized trainer.
        """
        return super()._initialize_trainer()

    def train(
        self, save_checkpoint: bool = True, load_checkpoint: bool = False
    ) -> None:
        """Train the model.

        Args:
            save_checkpoint (bool, optional): Whether to save the checkpoint.
                Defaults to True.
            load_checkpoint (bool, optional): Whether to load the checkpoint.
                Defaults to False.

        Returns:
            None
        """
        self._trainer.train(self._model,
                            load_checkpoint=load_checkpoint,
                            save_checkpoint=save_checkpoint)

    def evaluate(self, plot_cm: bool = False) -> dict:
        """Evaluate the model.

        Args:
            plot_cm (bool): Whether to plot the confusion matrix.
                Defaults to False.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        return super().evaluate(plot_cm)
