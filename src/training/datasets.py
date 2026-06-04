"""
PyTorch dataset and dataloader utilities.

This module converts windowed NumPy arrays into PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.windowing import SequenceWindowData


class SequenceTensorDataset(Dataset):
    """
    Dataset for many-to-one time-series binary classification.

    Each item contains:
    - X: shape (sequence_length, feature_count)
    - y: scalar binary target
    """

    def __init__(self, window_data: SequenceWindowData) -> None:
        self.features = torch.as_tensor(
            window_data.X,
            dtype=torch.float32
        )
        self.targets = torch.as_tensor(
            window_data.y,
            dtype=torch.float32
        )

        if self.features.ndim != 3:
            raise ValueError(
                "Window features must have shape "
                "(samples, sequence_length, feature_count)."
            )

        if self.targets.ndim != 1:
            raise ValueError("Window targets must be one-dimensional.")

        if len(self.features) != len(self.targets):
            raise ValueError(
                "Feature and target sample counts do not match."
            )

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def create_data_loaders(
    windowed_partitions: Dict[str, SequenceWindowData],
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create train, validation and test DataLoaders.

    Rules
    -----
    - Train loader is shuffled.
    - Validation and test loaders preserve ordering.
    - pin_memory is enabled only when CUDA is used.
    """
    required_partitions = {"train", "validation", "test"}
    missing_partitions = required_partitions.difference(
        windowed_partitions.keys()
    )

    if missing_partitions:
        raise ValueError(
            f"Missing windowed partitions: {sorted(missing_partitions)}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = device.type == "cuda"

    datasets = {
        partition: SequenceTensorDataset(window_data)
        for partition, window_data in windowed_partitions.items()
    }

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }