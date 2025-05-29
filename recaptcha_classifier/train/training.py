import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval import metrics
from tqdm import tqdm
from torch import nn, optim


class Trainer(object):
    """Contains main training logic"""

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 epochs: int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 save_folder: str,
                 model_file_name: str ='model.pt',
                 optimizer_file_name: str ='optimizer.pt',
                 scheduler_file_name: str ='scheduler.pt',
                 device: torch.device |None = None,
                 early_stop_threshold: int = 5
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
        self.save_folder = save_folder
        self.model_file_name = model_file_name
        self.optimizer_file_name = optimizer_file_name
        self.scheduler_file_name = scheduler_file_name

        self.device = None
        self.select_device(device)

        self._loss_acc_history = []
        self.optimizer = None
        self.scheduler = None
        
        self._early_stop_threshold = early_stop_threshold
        self._best_val_loss = float('inf')
        self._stagnation_counter = 0


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


    def train(self,
              model: nn.Module,
              lr: float = 0.001,
              load_checkpoint: bool = False,
              save_checkpoint: bool = True) -> None:
        """
        Main training loop. Uses RAdam as optimizer and StepLR as scheduler.
        :param lr: learning rate.
        :param save_checkpoint: If True, saves checkpoint files.
        :param model: Model for training.
        :param load_checkpoint: If true, loads latest checkpoint if it exists.
        with the previous states of the model, optimizer, and scheduler.
        """
        if not load_checkpoint:
            self._reset_trainer_state(model, lr)

        os.makedirs(self.save_folder, exist_ok=True)
        
        start_epoch = 0

        if load_checkpoint and os.path.exists(os.path.join(self.save_folder, self.model_file_name)):
            start_epoch = self.load_checkpoint_states(model)

        if start_epoch == 0:
            self._loss_acc_history = []

        model.to(self.device)
        print(f"Using device: {self.device}")

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

            val_loss = val_loss_counter.compute().item()
            val_acc = val_accuracy_counter.compute().item()
            print(f"Epoch {epoch+1} - Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

            if self._early_stop(val_loss):
                print(f"Early stopping at epoch {epoch + 1}.")
                print(f"Best validation loss: {self._best_val_loss:.4f}")
                break


    def _train_one_epoch(self, model, train_accuracy_counter, train_loss_counter, train_progress_bar):
        model.train()
        for data, targets in train_progress_bar:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = model(data)
            loss = nn.functional.cross_entropy(predictions, targets)
            loss.backward()
            self.optimizer.step()

            train_accuracy_counter.update(predictions, targets)
            train_loss_counter.update(loss, weight=data.size(0))
            
            batch_loss = loss.item()
            batch_acc = (predictions.argmax(dim=1) == targets).float().mean().item()

            train_progress_bar.set_postfix(
                loss=batch_loss,
                accuracy=batch_acc
            )
        
        self._loss_acc_history.append([
            train_loss_counter.compute().item(),
            train_accuracy_counter.compute().item()
        ])

        self.scheduler.step()


    def _val_one_epoch(self, model, val_accuracy_counter, val_loss_counter, val_progress_bar):
        model.eval()
        with torch.no_grad():
            for data, targets in val_progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)

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
        if not os.path.exists(os.path.join(self.save_folder, self.model_file_name)):
            raise FileNotFoundError("Checkpoint folder doesn't exist.")

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

    def _reset_trainer_state(self, model: nn.Module, lr: float) -> None:
        """
        Resets the trainer state to prevent leakage between runs.
        :param model: Model to reset.
        :param lr: Learning rate for the optimizer.
        """
        self._loss_acc_history = []
        self.optimizer = optim.RAdam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self._best_val_loss = float('inf')
        self._stagnation_counter = 0
    
    def _early_stop(self, val_loss: float) -> bool:
        """
        Checks if early stopping criteria are met.
        :param val_loss: Current validation loss.
        :return: True if early stopping criteria are met, False otherwise.
        """
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._stagnation_counter = 0
            return False
        
        self._stagnation_counter += 1
        if self._stagnation_counter >= self._early_stop_threshold:
            print(f"Early stopping triggered after {self._stagnation_counter} epochs without improvement.")
            return True
        
        return False