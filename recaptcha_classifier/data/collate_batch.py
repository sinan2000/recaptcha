from typing import List
from .types import DataBatch, DataItem
import torch


def collate_batch(batch: List[DataItem]) -> DataBatch:
    """
    Custom collate function used in the PyTorch DataLoader to handle
    the non-uniform length of bounding box lists.

    Args:
        batch (List[DataItem]): A batch of training items; each item is
        a tuple of format (image tensor, class index).

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
