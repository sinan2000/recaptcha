import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from recaptcha_classifier.models.simple_classifier_model import SimpleCNN
from recaptcha_classifier.pipeline.base_pipeline import BasePipeline
from recaptcha_classifier.train.training import Trainer


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

    def _initialize_model(self) -> SimpleCNN:
        return SimpleCNN(num_classes=self.class_map_length)

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
        self._model = self._initialize_model()  # not same in main_model
        self.optimizer = optim.RAdam(self._model.parameters(), lr=self.lr)
        self.scheduler = StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma)
        self._trainer = self._initialize_trainer()
        self._trainer.train(model=self._model)
        self.evaluate()
