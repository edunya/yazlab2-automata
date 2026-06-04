"""
Tests for original-scenario deep learning experiment orchestration.

These tests use small synthetic arrays, one epoch and CPU execution.
They do not run full SKAB/BATADAL experiments or load GPU-heavy data.
"""

from copy import deepcopy
import json

import numpy as np
import pandas as pd

from src.data.splitting import DatasetSplit
from src.training.run_experiment import (
    prepare_deep_learning_partitions,
    run_deep_learning_original_split
)
from src.utils.config_loader import load_config
from src.utils.logger import ExperimentLogger


def make_runner_config():
    config = deepcopy(load_config())

    config["device"]["type"] = "cpu"
    config["training"]["max_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    config["training"]["batch_size"] = 8
    config["training"]["use_amp"] = False
    config["windowing"]["sequence_length"] = 4

    return config


def make_synthetic_split_data():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(36, 3)),
        columns=["sensor_a", "sensor_b", "sensor_c"]
    )

    y = pd.Series(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
        dtype=int
    )

    split = DatasetSplit(
        train_indices=np.arange(0, 16),
        validation_indices=np.arange(16, 26),
        test_indices=np.arange(26, 36),
        split_name="synthetic_original_split"
    )

    return X, y, split


def test_prepare_partitions_fits_train_only_scaler_and_windows_after_split():
    config = make_runner_config()
    X, y, split = make_synthetic_split_data()

    partitions, preprocessor = prepare_deep_learning_partitions(
        X=X,
        y=y,
        split=split,
        config=config
    )

    assert partitions["train"].X.shape == (13, 4, 3)
    assert partitions["validation"].X.shape == (7, 4, 3)
    assert partitions["test"].X.shape == (7, 4, 3)

    expected_train_mean = X.iloc[split.train_indices].mean().to_numpy()

    assert np.allclose(preprocessor.scaler.mean_, expected_train_mean)
    assert partitions["test"].target_indices.tolist() == [
        29, 30, 31, 32, 33, 34, 35
    ]


def test_run_original_deep_learning_split_returns_metrics_and_runtime():
    config = make_runner_config()
    X, y, split = make_synthetic_split_data()

    result = run_deep_learning_original_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC",
        model_name="gru",
        seed=42,
        device="cpu"
    )

    assert result.dataset == "SYNTHETIC"
    assert result.model_name == "gru"
    assert result.scenario == "original"
    assert result.seed == 42

    assert result.window_counts == {
        "train": 13,
        "validation": 7,
        "test": 7
    }

    assert result.evaluation_result.sample_count == 7
    assert result.test_scores.shape == (7,)
    assert result.test_labels.shape == (7,)

    assert result.runtime_record.training_seconds >= 0
    assert result.runtime_record.inference_seconds >= 0

    assert result.training_result.completed_epochs == 1
    assert result.training_result.device == "cpu"


def test_original_runner_can_save_json_and_csv_log_artifacts(tmp_path):
    config = make_runner_config()
    X, y, split = make_synthetic_split_data()

    logger = ExperimentLogger(
        log_dir=tmp_path,
        experiment_name="runner_smoke_test"
    )

    result = run_deep_learning_original_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC",
        model_name="cnn1d",
        seed=42,
        device="cpu",
        logger=logger
    )

    experiment_dir = tmp_path / "runner_smoke_test"

    assert (experiment_dir / "params.json").exists()
    assert (experiment_dir / "metrics.csv").exists()
    assert (experiment_dir / "training_history.json").exists()
    assert (experiment_dir / "summary.json").exists()

    summary = json.loads(
        (experiment_dir / "summary.json").read_text(encoding="utf-8")
    )

    assert summary["dataset"] == "SYNTHETIC"
    assert summary["model"] == "cnn1d"
    assert summary["test_metrics"]["sample_count"] == 7
    assert result.evaluation_result.sample_count == 7