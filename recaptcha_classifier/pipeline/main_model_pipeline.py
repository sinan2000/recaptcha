import os
import torch
from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.pipeline.base_pipeline import BasePipeline
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.kfold_validation import (
    KFoldValidation
)
from recaptcha_classifier.constants import (
    MODELS_FOLDER, MAIN_MODEL_FILE_NAME,
    OPTIMIZER_FILE_NAME, SCHEDULER_FILE_NAME
)

class MainClassifierPipeline(BasePipeline):
    """Pipeline for training a simple classifier model."""
    def __init__(self,
                 lr: float = 0.001,
                 k_folds: int = 5,
                 epochs: int = 15,
                 device: torch.device | None = None,
                 save_folder: str = MODELS_FOLDER,
                 model_file_name: str = MAIN_MODEL_FILE_NAME,
                 optimizer_file_name: str = OPTIMIZER_FILE_NAME,
                 scheduler_file_name: str = SCHEDULER_FILE_NAME
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
        super().__init__(lr, epochs, device,
                         save_folder, model_file_name,
                         optimizer_file_name, scheduler_file_name)

        self.k_folds = k_folds
        self._hp_optimizer = None

    def run(self,
            save_train_checkpoints: bool = True,
            load_train_checkpoints: bool = False) -> None:
        """ Runs the pipeline."""
        self._data_loader()
        self._trainer = self._initialize_trainer()
        self._hp_optimizer = HPOptimizer(trainer=self._trainer)

        print("~~ Hyperparameter optimization (Random Search) ~~")
        self._hp_optimizer.optimize_hyperparameters(
            save_checkpoints=save_train_checkpoints)

        # model gets initialized inside here:
        self._run_kfold_cross_validation()

        self._trainer.train(self._model,
                            self.lr,
                            save_checkpoint=save_train_checkpoints,
                            load_checkpoint=load_train_checkpoints)
        self.evaluate(plot_cm=True)

    def _run_kfold_cross_validation(self) -> None:
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
            train_loader=self._loaders["train"],
            val_loader=self._loaders["val"],
            k_folds=self.k_folds,
            device=self.device
        )

        best_hp = self._hp_optimizer.get_best_hp()

        self._kfold.run_cross_validation(hp=best_hp)

        self.lr = best_hp[2]
        self._model = self._initialize_model(
            n_layers=best_hp[0],
            kernel_size=best_hp[1])

    def _initialize_model(self, n_layers: int, kernel_size: int) -> MainCNN:
        return MainCNN(
            n_layers=n_layers, kernel_size=kernel_size,
            num_classes=self.class_map_length)

    def save_model(self):
        os.makedirs(self.save_folder, exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "config": {
                "n_layers": self._model.n_layers,
                "kernel_size": self._model.kernel_size,
            }
        }, os.path.join(self.save_folder, self.model_file_name))