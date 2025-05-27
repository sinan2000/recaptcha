import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from recaptcha_classifier.models.main_model.model_class import MainCNN
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.data.pipeline import DataPreprocessingPipeline
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model
from recaptcha_classifier.models.main_model.HPoptimizer import HPOptimizer
from recaptcha_classifier.models.main_model.kfold_validation import KFoldValidation


class SimpleClassifierPipeline:
    def __init__(self,
                 k_folds: int = 5,
                 n_layers: int = 3,
                 kernel_size: int = 5,
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

        self._model = MainCNN(n_layers=n_layers, 
                              kernel_size=kernel_size, 
                              num_classes=len(self.class_map))
        
        self._trainer = Trainer(train_loader=None,  # will be replaced per fold
                                val_loader=None,
                                epochs=self.epochs,
                                optimizer=None,     # set during HP
                                scheduler=None,    # hardcode???
                                save_folder=self.save_folder,
                                model_file_name=self.model_file_name,
                                optimizer_file_name=self.optimizer_file_name,
                                scheduler_file_name=self.scheduler_file_name,
                                device=self.device)

        self._hp_optimizer = HPOptimizer(trainer=self._trainer)
        self._kfold = KFoldValidation(
            data=self._data,
            k_folds=self.k_folds,
            hp_optimizer=self._hp_optimizer,
            device=self.device
        )
        
        
        # ????????
        self._load_checkpoint = (
            os.path.exists(os.path.join(save_folder, model_file_name)) and
            os.path.exists(os.path.join(save_folder, optimizer_file_name)) and
            os.path.exists(os.path.join(save_folder, scheduler_file_name))
        )

    def train(self, save_checkpoints: bool = True, top_n_models: int = 1) -> None:
        self._kfold.run_cross_validation(top_n_models=top_n_models, save_checkpoints=save_checkpoints)
        best_model, best_metrics = self._kfold.get_best_overall_model(metric_key='F1-score')
        self._loaders = self._data.run()
        self._trainer.train_loader = self._loaders["train"]
        self._trainer.val_loader = self._loaders["val"]

        # Re-initialize optimizer and scheduler for best model
        self._trainer.optimizer = optim.RAdam(best_model.parameters(), lr=self._trainer.optimizer.param_groups[0]['lr'])
        self._trainer.scheduler = StepLR(self._trainer.optimizer, step_size=5, gamma=0.5)
        self._trainer.train(best_model, self._load_checkpoint) # best model?? / where use best metrics

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
