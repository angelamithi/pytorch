from torch.utils.data import DataLoader
from torchvision import datasets
import os
from typing import Optional, Tuple, Any


def create_dataloaders(
    train_data=None,
    test_data=None,
    train_dir: str = None,
    test_dir: str = None,
    transform=None,
    batch_size: int = 32,
    num_workers: int = os.cpu_count()
):
    """
    Creates DataLoaders from either:
    1. Pre-built datasets (e.g. Food101)
    2. Image folders (train_dir/test_dir)

    Args:
        train_data: Optional prebuilt training dataset
        test_data: Optional prebuilt test dataset
        train_dir: Optional path for ImageFolder training data
        test_dir: Optional path for ImageFolder test data
        transform: torchvision transforms (used only if using ImageFolder)
        batch_size: batch size
        num_workers: dataloader workers

    Returns:
        train_dataloader, test_dataloader, class_names
    """

    # -------------------------------------------------------
    # CASE 1: If dataset objects are already provided
    # -------------------------------------------------------
    if train_data is not None and test_data is not None:
        class_names = getattr(train_data, "classes", None)

    # -------------------------------------------------------
    # CASE 2: If using ImageFolder paths
    # -------------------------------------------------------
    else:
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        class_names = train_data.classes

    # -------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names