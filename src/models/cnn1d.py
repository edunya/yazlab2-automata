"""
1D-CNN model for time-series anomaly classification.

The model learns local temporal patterns from multivariate sequences.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    """
    Many-to-one 1D-CNN binary classifier.

    Input shape
    -----------
    (batch_size, sequence_length, input_size)

    Internally Conv1d receives:
    (batch_size, input_size, sequence_length)

    Output shape
    ------------
    (batch_size,)

    Output values are raw logits for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_size: int,
        conv_channels: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be greater than zero.")

        if len(conv_channels) != 2:
            raise ValueError(
                "This lightweight CNN architecture expects exactly "
                "two convolution channel sizes."
            )

        if any(channel <= 0 for channel in conv_channels):
            raise ValueError("All convolution channel sizes must be positive.")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_size = input_size

        first_channels, second_channels = conv_channels
        padding = kernel_size // 2

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=first_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(first_channels),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=first_channels,
                out_channels=second_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(second_channels),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(second_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce one raw anomaly logit for each sequence.
        """
        if x.ndim != 3:
            raise ValueError(
                "CNN1D input must have shape "
                "(batch_size, sequence_length, input_size)."
            )

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, "
                f"received {x.shape[-1]}."
            )

        # Conv1d expects channels before the temporal dimension.
        x = x.transpose(1, 2)

        extracted_features = self.feature_extractor(x)
        logits = self.classifier(extracted_features)

        return logits.squeeze(-1)