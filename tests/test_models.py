"""
Tests for PyTorch deep learning model architectures.
"""

import pytest
import torch
from torch import nn

from src.models.cnn1d import CNN1DClassifier
from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier
from src.models.model_factory import (
    build_deep_learning_models,
    build_model,
    count_trainable_parameters
)
from src.utils.config_loader import load_config


@pytest.mark.parametrize("input_size", [8, 43])
def test_lstm_output_shape(input_size: int):
    model = LSTMClassifier(input_size=input_size)
    X = torch.randn(4, 32, input_size)

    logits = model(X)

    assert logits.shape == (4,)


@pytest.mark.parametrize("input_size", [8, 43])
def test_gru_output_shape(input_size: int):
    model = GRUClassifier(input_size=input_size)
    X = torch.randn(4, 32, input_size)

    logits = model(X)

    assert logits.shape == (4,)


@pytest.mark.parametrize("input_size", [8, 43])
def test_cnn1d_output_shape(input_size: int):
    model = CNN1DClassifier(input_size=input_size)
    X = torch.randn(4, 32, input_size)

    logits = model(X)

    assert logits.shape == (4,)


@pytest.mark.parametrize("model_name", ["lstm", "gru", "cnn1d"])
def test_model_factory_builds_supported_models(model_name: str):
    config = load_config()

    model = build_model(
        model_name=model_name,
        input_size=8,
        config=config
    )

    assert isinstance(model, nn.Module)
    assert count_trainable_parameters(model) > 0


def test_build_all_enabled_deep_learning_models_excludes_automata():
    config = load_config()

    models = build_deep_learning_models(
        input_size=8,
        config=config
    )

    assert set(models.keys()) == {"lstm", "gru", "cnn1d"}
    assert "automata" not in models


@pytest.mark.parametrize("model_name", ["lstm", "gru", "cnn1d"])
def test_models_are_compatible_with_binary_logit_loss(model_name: str):
    config = load_config()

    model = build_model(
        model_name=model_name,
        input_size=8,
        config=config
    )

    X = torch.randn(4, 32, 8)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    logits = model(X)
    loss = nn.BCEWithLogitsLoss()(logits, y)

    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_factory_rejects_unknown_model():
    config = load_config()

    with pytest.raises(ValueError):
        build_model(
            model_name="transformer",
            input_size=8,
            config=config
        )


@pytest.mark.parametrize(
    "model_class",
    [LSTMClassifier, GRUClassifier, CNN1DClassifier]
)
def test_models_reject_invalid_input_shape(model_class):
    model = model_class(input_size=8)
    invalid_X = torch.randn(4, 8)

    with pytest.raises(ValueError):
        model(invalid_X)