"""
LSTM model for time-series anomaly classification.

The model receives a sequence of observations and predicts
the anomaly label of the final time step.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """
    Many-to-one LSTM binary classifier.

    Input shape
    -----------
    (batch_size, sequence_length, input_size)

    Output shape
    ------------
    (batch_size,)

    Output values are raw logits. Sigmoid is not applied inside the model
    because training will use BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be greater than zero.")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")

        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # PyTorch applies internal recurrent dropout only between stacked
        # recurrent layers. For a single-layer LSTM this must be zero.
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        representation_size = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(representation_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce one raw anomaly logit for each sequence.
        """
        if x.ndim != 3:
            raise ValueError(
                "LSTM input must have shape "
                "(batch_size, sequence_length, input_size)."
            )

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, "
                f"received {x.shape[-1]}."
            )

        _, (hidden_state, _) = self.lstm(x)

        if self.bidirectional:
            sequence_representation = torch.cat(
                (hidden_state[-2], hidden_state[-1]),
                dim=1
            )
        else:
            sequence_representation = hidden_state[-1]

        logits = self.classifier(sequence_representation)

        return logits.squeeze(-1)