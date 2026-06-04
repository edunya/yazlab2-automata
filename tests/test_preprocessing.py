"""
Tests for preprocessing utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    AutomataPreprocessor,
    DeepLearningPreprocessor,
    build_automata_preprocessor_from_config
)
from src.utils.config_loader import load_config


def test_deep_learning_preprocessor_fits_only_training_distribution():
    X_train = pd.DataFrame({
        "sensor_a": [0.0, 1.0, 2.0],
        "sensor_b": [10.0, 11.0, 12.0]
    })

    X_test = pd.DataFrame({
        "sensor_a": [100.0, 101.0],
        "sensor_b": [110.0, 111.0]
    })

    preprocessor = DeepLearningPreprocessor()
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    assert np.allclose(transformed_train.mean().to_numpy(), [0.0, 0.0])
    assert not np.allclose(transformed_test.mean().to_numpy(), [0.0, 0.0])


def test_deep_learning_preprocessor_preserves_shape_and_index():
    X_train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0]
        },
        index=[10, 11, 12]
    )

    preprocessor = DeepLearningPreprocessor()
    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape == X_train.shape
    assert transformed.index.tolist() == [10, 11, 12]
    assert transformed.columns.tolist() == ["a", "b"]


def test_deep_learning_preprocessor_rejects_changed_columns():
    X_train = pd.DataFrame({
        "a": [1.0, 2.0],
        "b": [3.0, 4.0]
    })

    X_invalid = pd.DataFrame({
        "b": [3.0, 4.0],
        "a": [1.0, 2.0]
    })

    preprocessor = DeepLearningPreprocessor().fit(X_train)

    with pytest.raises(ValueError):
        preprocessor.transform(X_invalid)


def test_automata_preprocessor_returns_single_pc1_column():
    X_train = pd.DataFrame({
        "sensor_a": [0.0, 1.0, 2.0, 3.0],
        "sensor_b": [2.0, 3.0, 4.0, 5.0],
        "sensor_c": [5.0, 4.0, 3.0, 2.0]
    })

    preprocessor = AutomataPreprocessor(n_components=1)
    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape == (4, 1)
    assert transformed.columns.tolist() == ["PC1"]
    assert len(preprocessor.pca.explained_variance_ratio_) == 1


def test_build_automata_preprocessor_from_config():
    config = load_config()

    preprocessor = build_automata_preprocessor_from_config(config)

    assert preprocessor.n_components == 1