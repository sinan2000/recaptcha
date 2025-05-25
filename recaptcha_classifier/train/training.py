import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval import metrics
from tqdm import tqdm
from torch import nn


class Trainer(object):
    """Contains main training logic"""

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 epochs: int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 save_folder: str,
                 model_file_name='model.pt',
                 optimizer_file_name='optimizer.pt',
                 scheduler_file_name='scheduler.pt',
                 device: torch.device |None = None
                 ):
        """
        Constructor for Trainer class.
        :param train_loader: Dataloader for training data.
        :param epochs: Number of epochs.
        :param save_folder: Folder for saving checkpoint files.
        :param optimizer: Optimizer for training.
        :param scheduler: Scheduler for training.
        :param model_file_name: Model checkpoint file name.
        :param optimizer_file_name: Optimizer checkpoint file name.
        :param scheduler_file_name: Scheduler checkpoint file name.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_folder = save_folder
        self.model_file_name = model_file_name
        self.optimizer_file_name = optimizer_file_name
        self.scheduler_file_name = scheduler_file_name

        self.device = None
        self.select_device(device)

        self._loss_acc_history = []


    @property
    def loss_acc_history(self) -> np.ndarray:
        return np.array(self._loss_acc_history.copy())


    def select_device(self, device=None):
        """
        Configures device for training.
        :param device: Selects device as specified. If None, uses cuda if available, else cpu.
        """
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def train(self, model, load_checkpoint: bool, save_checkpoint: bool = True) -> None:
        """
        Main training loop.
        :param save_checkpoint: If True, saves checkpoint files.
        :param model: Model for training.
        :param load_checkpoint: If true, loads latest checkpoint if it exists.
        with the previous states of the model, optimizer, and scheduler.
        """

        os.makedirs(self.save_folder, exist_ok=True)
        start_epoch = 0
        if load_checkpoint and os.path.exists(os.path.join(self.save_folder, self.model_file_name)):
            start_epoch = self.load_checkpoint_states(model)

        if start_epoch == 0:
            self._loss_acc_history = []

        model.to(self.device)
        model.train()

        for epoch in range(start_epoch, self.epochs):
            # Training
            train_accuracy_counter = metrics.MulticlassAccuracy().to(self.device)
            train_loss_counter = metrics.Mean().to(self.device)

            train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            self._train_one_epoch(model,
                                      train_accuracy_counter,
                                      train_loss_counter,
                                      train_progress_bar)

            # Saving states
            if save_checkpoint:
                self.save_checkpoint_states(model)

            # Validation
            val_accuracy_counter = metrics.MulticlassAccuracy().to(self.device)
            val_loss_counter = metrics.Mean().to(self.device)

            val_progress_bar = tqdm(self.val_loader, desc="Eval")
            self._val_one_epoch(model,
                                    val_accuracy_counter,
                                    val_loss_counter,
                                    val_progress_bar)


    def _train_one_epoch(self, model, train_accuracy_counter, train_loss_counter, train_progress_bar):
        for data, targets in train_progress_bar:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = model(data)
            loss = nn.functional.cross_entropy(predictions, targets)
            loss.backward()
            self.optimizer.step()

            train_accuracy_counter.update(predictions, targets)
            train_loss_counter.update(loss, weight=data.size(0))

            loss_item = train_loss_counter.compute().item()
            acc_item = train_accuracy_counter.compute().item()
            train_progress_bar.set_postfix(
                loss=loss_item,
                accuracy=acc_item
            )
            self._loss_acc_history.append([loss_item, acc_item])

        self.scheduler.step()


    def _val_one_epoch(self, model, val_accuracy_counter, val_loss_counter, val_progress_bar):
        for data, targets in val_progress_bar:
            data, targets = data.to(self.device), targets.to(self.device)

            model.eval()
            with torch.no_grad():
                predictions = model(data)
                loss = nn.functional.cross_entropy(predictions, targets)

                val_accuracy_counter.update(predictions, targets)
                val_loss_counter.update(loss, weight=data.size(0))

                val_progress_bar.set_postfix(
                    loss=val_loss_counter.compute().item(),
                    accuracy=val_accuracy_counter.compute().item()
                )


    def save_checkpoint_states(self, model) -> None:
        """
        Saves states of model, optimizer, and scheduler.
        """
        torch.save(model.state_dict(), os.path.join(self.save_folder, self.model_file_name))
        torch.save(self.optimizer.state_dict(), os.path.join(self.save_folder, self.optimizer_file_name))
        torch.save(self.scheduler.state_dict(), os.path.join(self.save_folder, self.scheduler_file_name))
        print("Saved checkpoint.")


    def load_checkpoint_states(self, model) -> int:
        """
        Loads states of model, optimizer, and scheduler.
        :return: epoch index from which the checkpoint states were saved.
        """
        checkpoint_model = torch.load(os.path.join(self.save_folder, self.model_file_name))
        model.load_state_dict(checkpoint_model)
        checkpoint_optimizer = torch.load(os.path.join(self.save_folder, self.optimizer_file_name))
        self.optimizer.load_state_dict(checkpoint_optimizer)
        checkpoint_scheduler = torch.load(os.path.join(self.save_folder, self.scheduler_file_name))
        self.scheduler.load_state_dict(checkpoint_scheduler)
        start_epoch = self.scheduler._step_count - 1
        return start_epoch

    def delete_checkpoints(self):
        if not os.path.exists(self.save_folder):
            return
        for filename in os.listdir(self.save_folder):
            file_path = os.path.join(self.save_folder, filename)

            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {filename}")
        
        if not os.listdir(self.save_folder): # make sure it's empty
          os.rmdir(self.save_folder)
          print(f"Deleted folder: {self.save_folder}")
