import torch
from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.kfold_validation import (
    KFoldValidation
)
from recaptcha_classifier.pipeline import BasePipeline


class SimpleClassifierPipeline(BasePipeline):
    """Pipeline for training a simple classifier model."""
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

        Returns:
            None
        """
        super().__init__(step_size, gamma, lr, epochs, device,
                         save_folder, model_file_name,
                         optimizer_file_name, scheduler_file_name)

        self._hp_optimizer = None

    def run(self) -> None:
        """ Runs the pipeline."""
        self.data_loader()
        self._trainer = self._initialize_trainer(self._model)
        self._hp_optimizer = HPOptimizer(trainer=self._trainer)

        # the best model gets initialized inside here
        self._run_kfold_cross_validation()

        self._trainer.train()
        self.evaluate()

    def data_loader(self) -> None:
        """Loads the data."""
        super().data_loader()

    def _run_kfold_cross_validation(self,
                                    k_folds: int = 5,
                                    n_layers: int = 3,
                                    kernel_size: int = 5) -> None:
        """
        Runs k-fold cross validation.

        Args:
            k_folds (int, optional): Number of folds. Defaults to 5.
            n_layers (int, optional): Number of layers in the model.
                Defaults to 3.
            kernel_size (int, optional): Kernel size for the model.
                Defaults to 5.

        Returns:
            None
        """
        self._kfold = KFoldValidation(
            data=self._data,
            k_folds=self.k_folds,
            hp_optimizer=self._hp_optimizer,
            device=self.device
        )
        self._kfold.run_cross_validation(save_checkpoints=True)
        best_model = self._kfold.get_best_overall_model(metric_key='F1-score')

        # need lr here?
        self._model = self._initialize_model(
            n_layers=int(best_model['n_layers']),
            kernel_size=int(best_model['kernel_size']))

    def _initialize_model(self, n_layers: int, kernel_size: int) -> MainCNN:
        return MainCNN(
            n_layers=n_layers, kernel_size=kernel_size,
            num_classes=self.class_map_length)

    def _initialize_trainer(self) -> Trainer:
        return super()._initialize_trainer()

    def train(
        self, save_checkpoint: bool = True, load_checkpoint: bool = False
    ) -> None:
        self._trainer.train(self._model,
                            load_checkpoint=load_checkpoint,
                            save_checkpoint=save_checkpoint)

    def evaluate(self, plot_cm: bool = False) -> dict:
        return super().evaluate(plot_cm)
