"""
Tests for original-scenario probabilistic automata experiment runner.

These tests use small synthetic datasets only.
No full SKAB or BATADAL experiment is executed.
"""

from copy import deepcopy
import json

import numpy as np
import pandas as pd

from src.automata.run_experiment import (
    PC1Sequence,
    create_paa_word_sequence,
    prepare_pc1_partition_sequences,
    run_automata_original_split
)
from src.data.splitting import DatasetSplit
from src.utils.config_loader import load_config
from src.utils.logger import ExperimentLogger


def make_automata_runner_config():
    config = deepcopy(load_config())

    config["automata"]["context_length"] = 4
    config["automata"]["default_window_size"] = 2
    config["automata"]["default_alphabet_size"] = 3
    config["automata"]["smoothing"] = 1e-3

    return config


def make_synthetic_automata_data():
    base_pattern = np.asarray([0.0, 1.0, 0.0, -1.0] * 18)

    X = pd.DataFrame({
        "sensor_a": base_pattern,
        "sensor_b": np.roll(base_pattern, 1)
    })

    y = pd.Series(np.zeros(len(X), dtype=int))

    # Validation anomalies.
    y.iloc[42:45] = 1

    # Test anomalies.
    y.iloc[62:65] = 1

    split = DatasetSplit(
        train_indices=np.arange(0, 32),
        validation_indices=np.arange(32, 52),
        test_indices=np.arange(52, 72),
        split_name="synthetic_automata_original_split"
    )

    return X, y, split


def test_paa_word_sequence_preserves_last_time_step_label_alignment():
    sequence = PC1Sequence(
        sequence_id="train::file_a",
        values=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        labels=np.asarray([0, 0, 1, 0, 1]),
        target_indices=np.asarray([10, 11, 12, 13, 14])
    )

    words = create_paa_word_sequence(
        sequence=sequence,
        context_length=4,
        word_size=2
    )

    assert words.paa_values.shape == (2, 2)
    assert np.allclose(words.paa_values[0], [1.5, 3.5])
    assert np.allclose(words.paa_values[1], [2.5, 4.5])

    assert words.labels.tolist() == [0, 1]
    assert words.target_indices.tolist() == [13, 14]


def test_pc1_sequences_respect_source_file_boundaries():
    pc1 = pd.DataFrame({"PC1": np.arange(24, dtype=float)})
    y = pd.Series(np.zeros(24, dtype=int))
    groups = pd.Series(["file_a"] * 12 + ["file_b"] * 12)

    split = DatasetSplit(
        train_indices=np.asarray([0, 1, 2, 3, 12, 13, 14, 15]),
        validation_indices=np.asarray([4, 5, 6, 7, 16, 17, 18, 19]),
        test_indices=np.asarray([8, 9, 10, 11, 20, 21, 22, 23]),
        split_name="group_boundary_test"
    )

    partitions = prepare_pc1_partition_sequences(
        pc1=pc1,
        y=y,
        split=split,
        groups=groups
    )

    assert set(partitions["train"].keys()) == {
        "train::file_a",
        "train::file_b"
    }

    assert partitions["train"]["train::file_a"].values.tolist() == [
        0.0, 1.0, 2.0, 3.0
    ]

    assert partitions["train"]["train::file_b"].values.tolist() == [
        12.0, 13.0, 14.0, 15.0
    ]


def test_original_automata_runner_returns_calibrated_metrics_and_explanations():
    config = make_automata_runner_config()
    X, y, split = make_synthetic_automata_data()

    result = run_automata_original_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC"
    )

    assert result.dataset == "SYNTHETIC"
    assert result.scenario == "original"
    assert result.fold is None

    assert result.state_counts == {
        "train": 29,
        "validation": 17,
        "test": 17
    }

    assert result.evaluation_result.sample_count == 16
    assert result.test_scores.shape == (16,)
    assert result.test_labels.shape == (16,)
    assert len(result.test_explanations) == 16

    assert result.calibration_result.validation_sample_count == 16
    assert result.runtime_record.training_seconds >= 0
    assert result.runtime_record.inference_seconds >= 0

    assert result.preprocessing_summary["pca_n_components"] == 1
    assert result.automata_summary["model"] == "probabilistic_automata"


def test_original_automata_runner_saves_json_and_csv_artifacts(tmp_path):
    config = make_automata_runner_config()
    X, y, split = make_synthetic_automata_data()

    logger = ExperimentLogger(
        log_dir=tmp_path,
        experiment_name="automata_runner_smoke_test"
    )

    result = run_automata_original_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC",
        logger=logger
    )

    experiment_dir = tmp_path / "automata_runner_smoke_test"

    assert (experiment_dir / "params.json").exists()
    assert (experiment_dir / "metrics.csv").exists()
    assert (experiment_dir / "threshold_calibration.json").exists()
    assert (experiment_dir / "test_explanations.json").exists()
    assert (experiment_dir / "summary.json").exists()

    summary = json.loads(
        (experiment_dir / "summary.json").read_text(encoding="utf-8")
    )

    explanations = json.loads(
        (experiment_dir / "test_explanations.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["dataset"] == "SYNTHETIC"
    assert summary["model"] == "automata"
    assert summary["test_metrics"]["sample_count"] == 16
    assert len(explanations["explanations"]) == 16
    assert result.evaluation_result.sample_count == 16