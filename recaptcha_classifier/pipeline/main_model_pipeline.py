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

        # self._model = self._initialize_model(self._class_map)
        self._trainer = self._initialize_trainer()
        self._hp_optimizer = HPOptimizer(trainer=self._trainer)
        self._run_kfold_cross_validation()

    def data_loader(self) -> None:
        super().data_loader()

    def _run_kfold_cross_validation(self,
                                    k_folds: int = 5,
                                    n_layers: int = 3,
                                    kernel_size: int = 5,
                                    top_n_models: int = 1) -> None:

        self._kfold = KFoldValidation(
            data=self._data,
            k_folds=self.k_folds,
            hp_optimizer=self._hp_optimizer,
            device=self.device
        )
        self._kfold.run_cross_validation(
            top_n_models=top_n_models, save_checkpoints=True)
        best_model = self._kfold.get_best_overall_model(metric_key='F1-score') # should have [n_layers, kernel_size, learning rate] - can be from the HPO pandas dataframe
        self._initialize_model(n_layers=int(best_model['n_layers']),
                               kernel_size=int(best_model['kernel_size']))
        self._best_models = self._kfold.get_all_best_models()

    def _initialize_model(self, n_layers: int, kernel_size: int) -> MainCNN:
        return MainCNN(
            n_layers=n_layers, kernel_size=kernel_size,
            num_classes=self.class_map_length)

    def _initialize_trainer(self) -> Trainer:
        return Trainer(train_loader=self._loaders["train"],  # will be replaced per fold???
                       val_loader=self._loaders["val"],  # ???
                       epochs=self.epochs,
                       optimizer=self.optimizer,
                       scheduler=self.scheduler,
                       save_folder=self.save_folder,
                       model_file_name=self.model_file_name,
                       optimizer_file_name=self.optimizer_file_name,
                       scheduler_file_name=self.scheduler_file_name,
                       device=self.device)

    def train(
        self, save_checkpoint: bool = True, load_checkpoint: bool = False
    ) -> None:
        self._trainer.train(self._model,
                            load_checkpoint=load_checkpoint,
                            save_checkpoint=save_checkpoint)

    def evaluate(self, plot_cm: bool = False) -> dict:
        return super().evaluate(plot_cm)
