from abc import abstractmethod
import torch
from recaptcha_classifier.detection_labels import DetectionLabels
from recaptcha_classifier.data.pipeline import DataPreprocessingPipeline
from recaptcha_classifier.train.training import Trainer
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
        self.step_size = step_size
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.save_folder = save_folder
        self.model_file_name = model_file_name
        self.optimizer_file_name = optimizer_file_name
        self.scheduler_file_name = scheduler_file_name

        self._class_map = DetectionLabels.to_class_map()
        self._class_map_length = self.class_map_length
        self._loaders = None
        self._data = None
        self._model = None
        self.optimizer = None
        self.scheduler = None
        self._trainer = None

    def data_loader(self):  # make private?
        if self._loaders is None:
            self._data = DataPreprocessingPipeline(
                self._class_map, balance=True)
            self._loaders = self._data.run()
            print("Data loaders built successfully.")

    @abstractmethod
    def _initialize_model(self):
        pass

    def _initialize_trainer(self) -> Trainer:
        return Trainer(train_loader=self._loaders["train"],
                       val_loader=self._loaders["val"],
                       epochs=self.epochs,
                       optimizer=self.optimizer,
                       scheduler=self.scheduler,
                       save_folder=self.save_folder,
                       model_file_name=self.model_file_name,
                       optimizer_file_name=self.optimizer_file_name,
                       scheduler_file_name=self.scheduler_file_name,
                       device=self.device)

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

    def run(self):  # abstract because the sequence of actions is
        # different for the two pipelines
        pass
