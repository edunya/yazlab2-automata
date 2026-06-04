"""
Tests for sample-level and automata artifact exports.

These tests use synthetic data only.
"""

from copy import deepcopy
import json

import numpy as np
import pandas as pd

from src.data.splitting import DatasetSplit
from src.experiments.artifact_export import (
    export_automata_robustness_artifacts,
    export_deep_learning_robustness_artifacts
)
from src.experiments.automata_robustness import (
    run_automata_gaussian_robustness_split
)
from src.experiments.deep_learning_robustness import (
    run_deep_learning_gaussian_robustness_split
)
from src.utils.config_loader import load_config


def make_dl_export_config():
    config = deepcopy(load_config())

    config["device"]["type"] = "cpu"
    config["training"]["max_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    config["training"]["batch_size"] = 8
    config["training"]["use_amp"] = False
    config["windowing"]["sequence_length"] = 4

    config["experiments"]["robustness"]["gaussian_noise"]["levels"] = [0.05]

    return config


def make_dl_data():
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
        split_name="synthetic_dl_export_split"
    )

    return X, y, split


def make_automata_export_config():
    config = deepcopy(load_config())

    config["automata"]["context_length"] = 4
    config["automata"]["default_window_size"] = 2
    config["automata"]["default_alphabet_size"] = 3
    config["automata"]["smoothing"] = 1e-3

    config["experiments"]["robustness"]["gaussian_noise"]["levels"] = [0.05]

    return config


def make_automata_data():
    base_pattern = np.asarray([0.0, 1.0, 0.0, -1.0] * 20)

    X = pd.DataFrame({
        "sensor_a": base_pattern,
        "sensor_b": np.roll(base_pattern, 1)
    })

    y = pd.Series(np.zeros(len(X), dtype=int))

    y.iloc[48:51] = 1
    y.iloc[70:73] = 1

    split = DatasetSplit(
        train_indices=np.arange(0, 36),
        validation_indices=np.arange(36, 60),
        test_indices=np.arange(60, 80),
        split_name="synthetic_automata_export_split"
    )

    return X, y, split


def test_deep_learning_artifact_export_saves_sample_predictions(tmp_path):
    config = make_dl_export_config()
    X, y, split = make_dl_data()

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

    exported = export_deep_learning_robustness_artifacts(
        task_id="dl_task",
        result=result,
        output_dir=tmp_path
    )

    task_dir = tmp_path / "dl_task"

    assert exported["task_summary"].exists()
    assert exported["training_history"].exists()
    assert (task_dir / "original" / "predictions.csv").exists()
    assert (
        task_dir / "gaussian_noise_0.05" / "predictions.csv"
    ).exists()

    predictions = pd.read_csv(
        task_dir / "original" / "predictions.csv"
    )

    assert set(predictions.columns) == {
        "target_index",
        "y_true",
        "score",
        "threshold",
        "y_pred"
    }

    assert predictions.shape[0] == 9


def test_automata_artifact_export_saves_explanations_and_transitions(tmp_path):
    config = make_automata_export_config()
    X, y, split = make_automata_data()

    result = run_automata_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC"
    )

    exported = export_automata_robustness_artifacts(
        task_id="automata_task",
        result=result,
        output_dir=tmp_path
    )

    task_dir = tmp_path / "automata_task"

    assert exported["transition_matrix"].exists()
    assert exported["observed_transition_edges"].exists()
    assert (task_dir / "original" / "predictions.csv").exists()
    assert (task_dir / "original" / "explanations.json").exists()

    explanations = json.loads(
        (task_dir / "original" / "explanations.json").read_text(
            encoding="utf-8"
        )
    )

    transition_matrix = pd.read_csv(
        task_dir / "transition_matrix.csv",
        index_col=0
    )

    observed_edges = pd.read_csv(
        task_dir / "observed_transition_edges.csv"
    )

    assert len(explanations["explanations"]) == 16
    assert not transition_matrix.empty
    assert not observed_edges.empty
    assert set(observed_edges.columns) == {
        "from_state",
        "to_state",
        "observed_count",
        "probability"
    }


def test_automata_scenario_result_retains_explanations_for_reporting():
    config = make_automata_export_config()
    X, y, split = make_automata_data()

    result = run_automata_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC"
    )

    original = result.scenario_results["original"]

    assert len(original.test_explanations) == 16
    assert (
        original.summary()["explanation_count"]
        == 16
    )