from typing import List
from .types import DataItem, DataBatch
import torch


def collate_batch(batch: List[DataItem]) -> DataBatch:
    """
    Custom collate function used in the PyTorch DataLoader to combine
    the list of data items into a stacked batch for model training.

    Args:
        batch (List[DataItem]): A list of dataset items, tensors of
        image and label pair.

    Returns:
        Batch: A single tuple containing:
            - images_tensor (Tensor): Batched images of
            shape (batch_size, 3, H, W)
            - labels_tensor (Tensor): A tensor of shape
            containing the class indices.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Stack them as (3, H, W) tensors
    images_tensor = torch.stack(images)
    labels_tensor = torch.stack(labels)

    return images_tensor, labels_tensor
