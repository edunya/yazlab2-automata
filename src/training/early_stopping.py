"""
Early stopping utility.

Training is stopped when validation loss does not improve for a configured
number of consecutive epochs.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import nn


@dataclass
class EarlyStopping:
    """
    Validation-loss based early stopping.

    Parameters
    ----------
    patience:
        Number of consecutive non-improving epochs allowed.

    min_delta:
        Minimum validation-loss decrease required to count as improvement.

    checkpoint_path:
        Optional path for saving the best model weights.
    """

    patience: int = 5
    min_delta: float = 0.0
    checkpoint_path: Optional[str | Path] = None

    best_loss: float = field(default=float("inf"), init=False)
    best_epoch: Optional[int] = field(default=None, init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)
    _best_state: Optional[dict] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be greater than zero.")

        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)

    def step(
        self,
        validation_loss: float,
        model: nn.Module,
        epoch: int
    ) -> bool:
        """
        Update early stopping state.

        Returns
        -------
        bool
            True when training should stop.
        """
        if not math.isfinite(validation_loss):
            raise ValueError("validation_loss must be finite.")

        improved = validation_loss < (self.best_loss - self.min_delta)

        if improved:
            self.best_loss = float(validation_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            self._best_state = copy.deepcopy(model.state_dict())

            if self.checkpoint_path is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._best_state, self.checkpoint_path)

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """
        Restore model weights from the best validation-loss epoch.
        """
        if self._best_state is None:
            raise RuntimeError("No best model state is available.")

        model.load_state_dict(self._best_state)