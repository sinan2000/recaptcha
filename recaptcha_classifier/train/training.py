import os
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
                 device: torch.Device|None = None
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


    def select_device(self, device=None):
        """
        Configures device for training.
        :param device: Selects device as specified. If None, uses cuda if available, else cpu.
        """
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def train(self, model, load_checkpoint: bool) -> None:
        """
        Main training loop.
        :param load_checkpoint: If true, loads latest checkpoint
        with the previous states of the model, optimizer, and scheduler.
        """

        os.makedirs(self.save_folder, exist_ok=True)
        start_epoch = 0
        if load_checkpoint and os.path.exists(os.path.join(self.save_folder, self.model_file_name)):
            start_epoch = self.load_checkpoint_states(model)


        model.to(self.device)
        model.train()

        for epoch in range(start_epoch, self.epochs):
            accuracy_counter = metrics.MulticlassAccuracy().to(self.device)
            loss_counter = metrics.Mean().to(self.device)

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                predictions = model(data)       # predictions?
                loss = nn.functional.cross_entropy(predictions, targets)
                loss.backward()
                self.optimizer.step()

                accuracy_counter.update(predictions, targets)
                loss_counter.update(loss, weight=data.size(0))

                progress_bar.set_postfix(
                    loss=loss_counter.compute().item(),
                    accuracy=accuracy_counter.compute().item()
                )
            self.scheduler.step()

            self.save_checkpoint_states(model)


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
