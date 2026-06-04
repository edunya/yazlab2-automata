"""
Model factory utilities.

This module creates PyTorch models from the central project configuration.
"""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from src.models.cnn1d import CNN1DClassifier
from src.models.gru import GRUClassifier
from src.models.lstm import LSTMClassifier


DEEP_LEARNING_MODELS = {"lstm", "gru", "cnn1d"}


def build_model(
    model_name: str,
    input_size: int,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Build one deep learning model from config.

    Parameters
    ----------
    model_name:
        Supported values: lstm, gru, cnn1d.

    input_size:
        Number of input features.
        - SKAB: 8
        - BATADAL: 43

    config:
        Project configuration dictionary.
    """
    normalized_name = model_name.lower().strip()

    if normalized_name not in DEEP_LEARNING_MODELS:
        raise ValueError(
            f"Unsupported deep learning model: {model_name}. "
            f"Supported values are {sorted(DEEP_LEARNING_MODELS)}."
        )

    architecture_configs = config["models"]["architectures"]
    model_config = architecture_configs[normalized_name]

    if normalized_name == "lstm":
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            bidirectional=bool(model_config["bidirectional"])
        )

    if normalized_name == "gru":
        return GRUClassifier(
            input_size=input_size,
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            bidirectional=bool(model_config["bidirectional"])
        )

    return CNN1DClassifier(
        input_size=input_size,
        conv_channels=tuple(model_config["conv_channels"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"])
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters of a PyTorch model.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def build_deep_learning_models(
    input_size: int,
    config: Dict[str, Any]
) -> Dict[str, nn.Module]:
    """
    Build all enabled deep learning models.

    Automata is intentionally excluded because it is not a PyTorch model.
    """
    enabled_models = config["models"]["enabled_models"]

    model_names = [
        model_name
        for model_name in enabled_models
        if model_name in DEEP_LEARNING_MODELS
    ]

    return {
        model_name: build_model(
            model_name=model_name,
            input_size=input_size,
            config=config
        )
        for model_name in model_names
    }