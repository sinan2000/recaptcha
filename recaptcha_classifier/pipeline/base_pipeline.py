from abc import abstractmethod
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.data.pipeline import DataPreprocessingPipeline
# from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.features.evaluation.evaluate import evaluate_model


class BasePipeline:
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
        self._class_map = DetectionLabels.to_class_map()
        self._class_map_length = len(self._class_map)
        self._loaders = None
        self._model = self._initialize_model()
        self.optimizer = optim.RAdam(self._model.parameters(), lr=lr)
        self.scheduler = StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)
        self._trainer = self._initialize_trainer()

    def data_loader(self):
        if self._loaders is None:
            self._data = DataPreprocessingPipeline(
                self._class_map, balance=True)
            self._loaders = self._data.run()
            print("Data loaders built successfully.")
        # return self._loaders  # we return???

    @abstractmethod
    def _initialize_model(self):
        pass

    @abstractmethod
    def _initialize_trainer(self):
        pass

    @property
    def class_map_length(self):
        return len(self._class_map)

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
