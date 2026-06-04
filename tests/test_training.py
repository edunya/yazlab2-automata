"""
Tests for PyTorch training utilities.

These tests use small synthetic tensors and run on CPU.
They do not start real dataset training or put load on the GPU.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.data.windowing import SequenceWindowData
from src.models.model_factory import build_model
from src.training.datasets import (
    SequenceTensorDataset,
    create_data_loaders
)
from src.training.early_stopping import EarlyStopping
from src.training.trainer import (
    compute_positive_class_weight,
    fit_model,
    predict_logits
)
from src.utils.config_loader import load_config
from src.utils.seed import set_seed


def make_synthetic_window_data(
    sample_count: int,
    feature_count: int = 8,
    sequence_length: int = 32
) -> SequenceWindowData:
    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(sample_count, sequence_length, feature_count)
    ).astype(np.float32)

    y = np.asarray(
        [0, 1] * (sample_count // 2),
        dtype=np.int64
    )

    if len(y) < sample_count:
        y = np.append(y, 0)

    return SequenceWindowData(
        X=X,
        y=y,
        target_indices=np.arange(sample_count)
    )


def test_sequence_tensor_dataset_shapes():
    window_data = make_synthetic_window_data(sample_count=10)

    dataset = SequenceTensorDataset(window_data)

    features, target = dataset[0]

    assert len(dataset) == 10
    assert features.shape == (32, 8)
    assert target.ndim == 0
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


def test_create_data_loaders():
    partitions = {
        "train": make_synthetic_window_data(32),
        "validation": make_synthetic_window_data(16),
        "test": make_synthetic_window_data(16)
    }

    loaders = create_data_loaders(
        windowed_partitions=partitions,
        batch_size=8,
        seed=42,
        device=torch.device("cpu"),
        num_workers=0
    )

    train_features, train_targets = next(iter(loaders["train"]))

    assert set(loaders.keys()) == {"train", "validation", "test"}
    assert train_features.shape == (8, 32, 8)
    assert train_targets.shape == (8,)


def test_positive_class_weight_uses_train_targets():
    train_targets = np.asarray([0, 0, 0, 1], dtype=np.int64)

    pos_weight = compute_positive_class_weight(train_targets)

    assert pos_weight == 3.0


def test_early_stopping_stops_after_patience():
    model = nn.Linear(2, 1)

    early_stopping = EarlyStopping(
        patience=2,
        min_delta=0.0
    )

    assert early_stopping.step(1.0, model, epoch=1) is False
    assert early_stopping.step(1.1, model, epoch=2) is False
    assert early_stopping.step(1.2, model, epoch=3) is True

    assert early_stopping.best_epoch == 1
    assert early_stopping.best_loss == 1.0


def test_fit_model_and_predict_logits_run_on_cpu():
    set_seed(42)

    config = deepcopy(load_config())

    config["training"]["max_epochs"] = 2
    config["training"]["early_stopping_patience"] = 2
    config["training"]["use_amp"] = False

    partitions = {
        "train": make_synthetic_window_data(32),
        "validation": make_synthetic_window_data(16),
        "test": make_synthetic_window_data(16)
    }

    loaders = create_data_loaders(
        windowed_partitions=partitions,
        batch_size=8,
        seed=42,
        device=torch.device("cpu"),
        num_workers=0
    )

    model = build_model(
        model_name="gru",
        input_size=8,
        config=config
    )

    result = fit_model(
        model=model,
        data_loaders=loaders,
        train_targets=partitions["train"].y,
        config=config,
        device="cpu"
    )

    logits, targets = predict_logits(
        model=model,
        data_loader=loaders["test"],
        device="cpu"
    )

    assert result.completed_epochs == 2
    assert result.device == "cpu"
    assert result.amp_enabled is False
    assert result.pos_weight == 1.0
    assert len(result.history) == 2

    assert logits.shape == (16,)
    assert targets.shape == (16,)