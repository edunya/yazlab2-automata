"""
PyTorch model training utilities.

This module provides:
- device selection
- class-weighted binary loss
- one-epoch training and validation
- AMP support for CUDA
- early stopping integration
- logit prediction for later evaluation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.early_stopping import EarlyStopping


@dataclass
class TrainingResult:
    """
    Summary returned after model fitting.
    """

    history: list[Dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    completed_epochs: int
    training_time_seconds: float
    device: str
    amp_enabled: bool
    pos_weight: Optional[float]


def resolve_device(config: Dict[str, Any]) -> torch.device:
    """
    Select CPU or CUDA according to config and hardware availability.
    """
    requested_device = config.get("device", {}).get("type", "auto").lower()

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was requested but CUDA is not available."
            )
        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "Unsupported device type. Expected 'auto', 'cuda' or 'cpu'."
    )


def compute_positive_class_weight(train_targets: np.ndarray) -> float:
    """
    Compute positive-class weight using training targets only.

    Formula
    -------
    negative_count / positive_count
    """
    targets = np.asarray(train_targets).astype(int)

    observed_labels = set(np.unique(targets).tolist())

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Targets must be binary, found labels: {observed_labels}"
        )

    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())

    if positive_count == 0:
        raise ValueError(
            "Cannot compute pos_weight: training set has no positive samples."
        )

    if negative_count == 0:
        raise ValueError(
            "Cannot compute pos_weight: training set has no negative samples."
        )

    return float(negative_count / positive_count)


def build_loss_function(
    train_targets: np.ndarray,
    config: Dict[str, Any],
    device: torch.device
) -> tuple[nn.Module, Optional[float]]:
    """
    Build binary classification loss according to config.
    """
    training_config = config["training"]

    if training_config["loss_function"] != "bce_with_logits":
        raise ValueError(
            "This project currently supports only BCEWithLogitsLoss."
        )

    strategy = training_config["class_imbalance_strategy"]

    if strategy == "train_pos_weight":
        pos_weight_value = compute_positive_class_weight(train_targets)

        pos_weight_tensor = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device
        )

        return (
            nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor),
            pos_weight_value
        )

    if strategy == "none":
        return nn.BCEWithLogitsLoss(), None

    raise ValueError(
        f"Unsupported class imbalance strategy: {strategy}"
    )


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Build optimizer from central configuration.
    """
    training_config = config["training"]
    optimizer_name = training_config["optimizer"].lower()
    learning_rate = float(training_config["learning_rate"])

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False
) -> float:
    """
    Run one training or evaluation epoch.

    If optimizer is given, the model is trained.
    Otherwise, the model is evaluated.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for features, targets in data_loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled
            ):
                logits = model(features)
                loss = criterion(logits, targets)

            if is_training:
                if amp_enabled:
                    if scaler is None:
                        raise RuntimeError(
                            "AMP is enabled but GradScaler is missing."
                        )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = int(features.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader did not provide any samples.")

    return total_loss / total_samples


def fit_model(
    model: nn.Module,
    data_loaders: Dict[str, DataLoader],
    train_targets: np.ndarray,
    config: Dict[str, Any],
    device: Optional[str | torch.device] = None,
    checkpoint_path: Optional[str | Path] = None
) -> TrainingResult:
    """
    Fit one model using validation-loss early stopping.

    Notes
    -----
    - This function trains one model on one pre-created split.
    - Full fold/seed experiment orchestration will be added later.
    """
    required_loaders = {"train", "validation"}
    missing_loaders = required_loaders.difference(data_loaders.keys())

    if missing_loaders:
        raise ValueError(
            f"Missing DataLoaders: {sorted(missing_loaders)}"
        )

    resolved_device = (
        torch.device(device)
        if device is not None
        else resolve_device(config)
    )

    model = model.to(resolved_device)

    criterion, pos_weight_value = build_loss_function(
        train_targets=train_targets,
        config=config,
        device=resolved_device
    )

    optimizer = build_optimizer(model, config)

    training_config = config["training"]

    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["early_stopping_patience"])
    min_delta = float(training_config["early_stopping_min_delta"])

    requested_amp = bool(training_config.get("use_amp", False))
    amp_enabled = requested_amp and resolved_device.type == "cuda"

    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_enabled
        else None
    )

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=checkpoint_path
    )

    history: list[Dict[str, float]] = []

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(
            model=model,
            data_loader=data_loaders["train"],
            criterion=criterion,
            device=resolved_device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled
        )

        validation_loss = run_epoch(
            model=model,
            data_loader=data_loaders["validation"],
            criterion=criterion,
            device=resolved_device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled
        )

        history.append({
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss)
        })

        if early_stopping.step(
            validation_loss=validation_loss,
            model=model,
            epoch=epoch
        ):
            break

    training_time_seconds = time.perf_counter() - start_time

    early_stopping.restore_best_weights(model)

    if early_stopping.best_epoch is None:
        raise RuntimeError("Training completed without a best epoch.")

    return TrainingResult(
        history=history,
        best_epoch=early_stopping.best_epoch,
        best_validation_loss=early_stopping.best_loss,
        stopped_early=early_stopping.should_stop,
        completed_epochs=len(history),
        training_time_seconds=float(training_time_seconds),
        device=str(resolved_device),
        amp_enabled=amp_enabled,
        pos_weight=pos_weight_value
    )


def predict_logits(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return raw logits and true labels for later metric calculation.
    """
    resolved_device = torch.device(device)

    model = model.to(resolved_device)
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(resolved_device, non_blocking=True)

            logits = model(features)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_targets)
    )
