import torch
from typing_extensions import override

from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.pipeline.base_pipeline import BasePipeline
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.kfold_validation import (
    KFoldValidation
)


class MainClassifierPipeline(BasePipeline):
    def __init__(self,
                 lr: float = 0.001,
                 k_folds: int = 5,
                 epochs: int = 15,
                 device: torch.device | None = None,
                 save_folder: str = "main_classifier_checkpoints",
                 model_file_name: str = "main_model.pt",
                 optimizer_file_name: str = "optimizer.pt",
                 scheduler_file_name: str = "scheduler.pt"
                 ):
        super().__init__(lr, epochs, device,
                         save_folder, model_file_name,
                         optimizer_file_name, scheduler_file_name)

        self.k_folds = k_folds
        self._hp_optimizer = None

    def run(self,
            save_train_checkpoints: bool = True,
            load_train_checkpoints: bool = False) -> None:
        self.data_loader()
        self._trainer = self._initialize_trainer()
        self._hp_optimizer = HPOptimizer(trainer=self._trainer)

        # model gets initialized inside here:
        self._run_kfold_cross_validation()

        self._trainer.train(self._model,
                            self.lr,
                            save_checkpoint=save_train_checkpoints,
                            load_checkpoint=load_train_checkpoints)
        self.evaluate()

    def data_loader(self) -> None:
        super().data_loader()

    def _run_kfold_cross_validation(self) -> None:
        self._kfold = KFoldValidation(
            train_loader=self._loaders["train"],
            val_loader=self._loaders["val"],
            k_folds=self.k_folds,
            hp_optimizer=self._hp_optimizer,
            device=self.device
        )
        self._kfold.run_cross_validation(save_checkpoints=True)
        best_model = self._kfold.get_best_overall_model(metric_key='F1-score')

        self.lr = best_model['lr']
        self._model = self._initialize_model(
            n_layers=int(best_model['n_layers']),
            kernel_size=int(best_model['kernel_size']))

    def _initialize_model(self, n_layers: int, kernel_size: int) -> MainCNN:
        return MainCNN(
            n_layers=n_layers, kernel_size=kernel_size,
            num_classes=self.class_map_length)

    def _initialize_trainer(self) -> Trainer:
        return super()._initialize_trainer()

    def evaluate(self, plot_cm: bool = False) -> dict:
        return super().evaluate(plot_cm)

    def save_model(self):
        super().save_model()
