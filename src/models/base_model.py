import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Base class for all models.
    Automatically provides a string summary of the model's
    parameters and constructor args.
    """
    def __init__(self) -> None:
        """Constructor for the BaseModel class.

        Returns:
            None
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def __str__(self) -> str:
        """Returns a string representation of the model's
        arguments and architecture."""
        summary = [f"{self.__class__.__name__} Model Summary:"]

        # constructor args
        init_args = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and isinstance(v, (
                int, float, str, tuple, list, bool))
        }
        if init_args:
            summary.append("\nArguments:")
            for k, v in init_args.items():
                summary.append(f"  {k:<20} = {v}")

        # List parameters
        summary.append("\nParameters:")
        for name, param in self.named_parameters():
            summary.append(f"  {name:<25} shape={tuple(param.shape)}")

        return "\n".join(summary)
