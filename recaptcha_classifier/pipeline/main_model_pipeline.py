import torch
from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.kfold_validation import (
    KFoldValidation
)
from recaptcha_classifier.pipeline import BasePipeline


class SimpleClassifierPipeline(BasePipeline):
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
        super().__init__(step_size, gamma, lr, epochs, device,
                         save_folder, model_file_name,
                         optimizer_file_name, scheduler_file_name)

    def data_loader(self) -> None:
        super().data_loader()

    def _run_kfold_cross_validation(self,
                                    k_folds: int = 5,
                                    n_layers: int = 3,
                                    kernel_size: int = 5) -> None:

        self._kfold = KFoldValidation(
            data=self._data,
            k_folds=self.k_folds,
            hp_optimizer=self._hp_optimizer,
            device=self.device
        )
        self._kfold.run_cross_validation(save_checkpoints=True)
        best_model = self._kfold.get_best_overall_model(metric_key='F1-score')

        # need lr here?
        self._initialize_model(n_layers=int(best_model['n_layers']),
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

    def run(self) -> None:
        self.data_loader()
        self._trainer = self._initialize_trainer(self._model)
        self._hp_optimizer = HPOptimizer(trainer=self._trainer)

        # model gets initialized inside here:
        self._run_kfold_cross_validation()

        self._trainer.train()
        self.evaluate()
