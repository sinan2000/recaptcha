import torch
from recaptcha_classifier.models.simple_classifier_model import SimpleCNN
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.pipeline import BasePipeline


class SimpleClassifierPipeline(BasePipeline):
    def __init__(self,
                 step_size: int = 5,
                 gamma: float = 0.5,
                 lr: float = 0.001,
                 epochs: int = 20,
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

    def _initialize_model(self) -> None:
        # call super??
        self._model = SimpleCNN(num_classes=self.class_map_length)
        # return model?

    def _initialize_trainer(self) -> None:
        # call super??
        self._trainer = Trainer(train_loader=self._loaders['train'],
                                val_loader=self._loaders['val'],
                                epochs=self.epochs,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                save_folder=self.save_folder,
                                model_file_name=self.model_file_name,
                                optimizer_file_name=self.optimizer_file_name,
                                scheduler_file_name=self.scheduler_file_name,
                                device=self.device)
        # return trainer??

    def train(
        self, save_checkpoint: bool = True, load_checkpoint: bool = False
    ) -> None:
        self._trainer.train(self._model,
                            load_checkpoint=load_checkpoint,
                            save_checkpoint=save_checkpoint)

    def evaluate(self, plot_cm: bool = False) -> dict:
        return super().evaluate(plot_cm)
