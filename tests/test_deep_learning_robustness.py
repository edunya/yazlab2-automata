"""
Tests for clean-model Gaussian-noise robustness evaluation.

These tests run on CPU, small synthetic data and one training epoch.
They do not launch full project experiments.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import DatasetSplit
from src.experiments.deep_learning_robustness import (
    run_deep_learning_gaussian_robustness_split
)
from src.utils.config_loader import load_config


def make_robustness_config():
    config = deepcopy(load_config())

    config["device"]["type"] = "cpu"
    config["training"]["max_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    config["training"]["batch_size"] = 8
    config["training"]["use_amp"] = False
    config["windowing"]["sequence_length"] = 4

    config["experiments"]["robustness"]["gaussian_noise"]["levels"] = [
        0.05,
        0.10
    ]

    return config


def make_synthetic_robustness_data():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(42, 3)),
        columns=["sensor_a", "sensor_b", "sensor_c"]
    )

    y = pd.Series(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
        dtype=int
    )

    split = DatasetSplit(
        train_indices=np.arange(0, 20),
        validation_indices=np.arange(20, 30),
        test_indices=np.arange(30, 42),
        split_name="synthetic_robustness_split"
    )

    return X, y, split


@pytest.mark.parametrize("model_name", ["gru", "cnn1d"])
def test_robustness_runner_evaluates_original_and_noisy_scenarios(model_name):
    config = make_robustness_config()
    X, y, split = make_synthetic_robustness_data()

    result = run_deep_learning_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC",
        model_name=model_name,
        seed=42,
        device="cpu"
    )

    assert result.dataset == "SYNTHETIC"
    assert result.model_name == model_name
    assert result.training_result.completed_epochs == 1

    assert set(result.scenario_results.keys()) == {
        "original",
        "gaussian_noise_0.05",
        "gaussian_noise_0.10"
    }

    assert result.clean_window_counts == {
        "train": 17,
        "validation": 7,
        "test": 9
    }

    for scenario_result in result.scenario_results.values():
        assert scenario_result.evaluation_result.sample_count == 9
        assert scenario_result.test_scores.shape == (9,)
        assert scenario_result.test_labels.shape == (9,)
        assert scenario_result.clean_model_reused is True


def test_noise_scenarios_do_not_retrain_or_change_target_alignment():
    config = make_robustness_config()
    X, y, split = make_synthetic_robustness_data()

    result = run_deep_learning_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC",
        model_name="gru",
        seed=42,
        device="cpu"
    )

    original = result.scenario_results["original"]
    noisy_005 = result.scenario_results["gaussian_noise_0.05"]
    noisy_010 = result.scenario_results["gaussian_noise_0.10"]

    assert original.runtime_record.training_seconds >= 0.0
    assert noisy_005.runtime_record.training_seconds == 0.0
    assert noisy_010.runtime_record.training_seconds == 0.0

    assert np.array_equal(
        original.test_target_indices,
        noisy_005.test_target_indices
    )
    assert np.array_equal(
        original.test_target_indices,
        noisy_010.test_target_indices
    )

    assert np.array_equal(
        original.test_labels,
        noisy_005.test_labels
    )
    assert np.array_equal(
        original.test_labels,
        noisy_010.test_labels
    )

    summary = result.summary()

    assert (
        summary["training_protocol"]["retrain_for_noise_levels"]
        is False
    )
    assert (
        summary["scenario_results"]["gaussian_noise_0.05"]
        ["clean_model_reused"]
        is True
    )