from abc import abstractmethod
import torch
from torch import nn
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.data.pipeline import DataPreprocessingPipeline
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model


class BasePipeline:
    """Base class for the two pipelines."""
    def __init__(self,
                 lr: float = 0.001,
                 epochs: int = 20,
                 device: torch.device | None = None,
                 save_folder: str = "checkpoints",
                 model_file_name: str = "model.pt",
                 optimizer_file_name: str = "optimizer.pt",
                 scheduler_file_name: str = "scheduler.pt",
                 early_stopping: bool = True
                 ) -> None:

        """
        Constructor for BasePipeline class.

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
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.save_folder = save_folder
        self.model_file_name = model_file_name
        self.optimizer_file_name = optimizer_file_name
        self.scheduler_file_name = scheduler_file_name
        self.early_stopping = early_stopping
        self._class_map = DetectionLabels
        self._loaders = None
        self._data = None
        self._model = None
        self._trainer = None

    @abstractmethod
    def run(self):
        """
        Run the pipeline.

        Returns:
            None
        """
        # different for the two pipelines
        pass

    def _data_loader(self) -> None:
        """ Load data using the DataPreprocessingPipeline.

        Returns:
            None
        """
        if self._loaders is None:
            self._data = DataPreprocessingPipeline(
                self._class_map, balance=True)
            self._loaders = self._data.run()
            print("Data loaders built successfully.")

    @abstractmethod
    def _initialize_model(self, **kwargs) -> nn.Module:
        """Initialize the model.

        Returns:
            nn.Module: The initialized model.
        """
        pass

    def _initialize_trainer(self) -> Trainer:
        """Initialize the trainer.

        Returns:
            Trainer: The initialized trainer.
        """
        return Trainer(train_loader=self._loaders["train"],
                       val_loader=self._loaders["val"],
                       epochs=self.epochs,
                       save_folder=self.save_folder,
                       model_file_name=self.model_file_name,
                       optimizer_file_name=self.optimizer_file_name,
                       scheduler_file_name=self.scheduler_file_name,
                       device=self.device,
                       early_stopping=self.early_stopping)

    @property
    def class_map_length(self) -> int:
        """Returns the length of the class map."""
        return len(self._class_map.all())

    def evaluate(self, plot_cm: bool = False) -> dict:
        """Evaluate the model.

        Args:
            plot_cm (bool, optional): Whether to plot the confusion matrix.
                Defaults to False.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        eval_results = evaluate_model(
            model=self._model,
            test_loader=self._loaders['test'],
            device=self._trainer.device,
            class_names=self._class_map.dataset_classnames(),
            plot_cm=plot_cm
        )
        return eval_results


    @abstractmethod
    def save_model(self):
        # different for the two pipelines; we need to
        # save n_layers and kernel_size for the main model
        pass