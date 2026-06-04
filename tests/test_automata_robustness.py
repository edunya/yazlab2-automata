"""
Tests for automata Gaussian-noise robustness and parameter-analysis runners.

These tests use small synthetic data only.
No complete SKAB or BATADAL experiment is executed.
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from src.data.splitting import DatasetSplit
from src.experiments.automata_robustness import (
    compute_automata_transition_structure,
    fit_clean_automata_pipeline,
    run_automata_gaussian_robustness_split,
    run_automata_parameter_analysis_split
)
from src.utils.config_loader import load_config


def make_automata_robustness_config():
    config = deepcopy(load_config())

    config["automata"]["context_length"] = 4
    config["automata"]["default_window_size"] = 2
    config["automata"]["default_alphabet_size"] = 3
    config["automata"]["smoothing"] = 1e-3

    config["experiments"]["robustness"]["gaussian_noise"]["levels"] = [
        0.05,
        0.10
    ]

    return config


def make_synthetic_automata_data():
    base_pattern = np.asarray([0.0, 1.0, 0.0, -1.0] * 20)

    X = pd.DataFrame({
        "sensor_a": base_pattern,
        "sensor_b": np.roll(base_pattern, 1)
    })

    y = pd.Series(np.zeros(len(X), dtype=int))

    # Validation must contain both classes for threshold calibration.
    y.iloc[48:51] = 1

    # Test anomalies.
    y.iloc[70:73] = 1

    split = DatasetSplit(
        train_indices=np.arange(0, 36),
        validation_indices=np.arange(36, 60),
        test_indices=np.arange(60, 80),
        split_name="synthetic_automata_robustness_split"
    )

    return X, y, split


def test_clean_fitted_pipeline_contains_calibrated_reusable_components():
    config = make_automata_robustness_config()
    X, y, split = make_synthetic_automata_data()

    fitted = fit_clean_automata_pipeline(
        X=X,
        y=y,
        split=split,
        config=config
    )

    assert fitted.preprocessor.is_fitted is True
    assert fitted.discretizer.is_fitted is True
    assert fitted.automata.is_fitted_ is True

    assert fitted.context_length == 4
    assert fitted.word_size == 2
    assert fitted.alphabet_size == 3

    assert fitted.calibration_result.validation_sample_count == 20
    assert fitted.training_seconds >= 0.0


def test_automata_robustness_reuses_clean_pipeline_for_noisy_test_scenarios():
    config = make_automata_robustness_config()
    X, y, split = make_synthetic_automata_data()

    result = run_automata_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC"
    )

    assert set(result.scenario_results.keys()) == {
        "original",
        "gaussian_noise_0.05",
        "gaussian_noise_0.10"
    }

    original = result.scenario_results["original"]
    noisy_005 = result.scenario_results["gaussian_noise_0.05"]
    noisy_010 = result.scenario_results["gaussian_noise_0.10"]

    assert original.clean_pipeline_reused is False
    assert noisy_005.clean_pipeline_reused is True
    assert noisy_010.clean_pipeline_reused is True

    assert original.runtime_record.training_seconds >= 0.0
    assert noisy_005.runtime_record.training_seconds == 0.0
    assert noisy_010.runtime_record.training_seconds == 0.0

    assert original.evaluation_result.sample_count == 16
    assert noisy_005.evaluation_result.sample_count == 16
    assert noisy_010.evaluation_result.sample_count == 16

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


def test_automata_robustness_produces_unseen_analysis_per_scenario():
    config = make_automata_robustness_config()
    X, y, split = make_synthetic_automata_data()

    result = run_automata_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC"
    )

    for scenario_result in result.scenario_results.values():
        unseen_summary = scenario_result.unseen_summary

        assert unseen_summary.total_decisions == 16
        assert (
            unseen_summary.seen_only_decisions
            + unseen_summary.unseen_involved_decisions
            == unseen_summary.total_decisions
        )

        assert 0.0 <= unseen_summary.unseen_decision_ratio <= 1.0
        assert 0.0 <= unseen_summary.unseen_state_occurrence_ratio <= 1.0

    summary = result.summary()

    assert (
        summary["protocol"]["refit_for_noise_levels"]
        is False
    )


def test_automata_parameter_analysis_returns_configured_grid_table():
    config = make_automata_robustness_config()
    X, y, split = make_synthetic_automata_data()

    config["experiments"]["parameter_analysis"]["automata"][
        "context_length_fixed"
    ] = 4

    config["experiments"]["parameter_analysis"]["automata"][
        "window_size_values"
    ] = [2, 3]

    config["experiments"]["parameter_analysis"]["automata"][
        "alphabet_size_values"
    ] = [3, 4]

    sweep_result = run_automata_parameter_analysis_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SYNTHETIC"
    )

    result_table = sweep_result.results_table()

    assert result_table.shape[0] == 4

    combinations = set(
        zip(
            result_table["window_size"],
            result_table["alphabet_size"]
        )
    )

    assert combinations == {
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 4)
    }

    assert "f1_score" in result_table.columns
    assert "state_count" in result_table.columns
    assert "training_seconds" in result_table.columns

    assert "observed_transition_count" in result_table.columns
    assert "possible_transition_count" in result_table.columns
    assert "transition_density" in result_table.columns

    assert result_table["observed_transition_count"].ge(1).all()
    assert result_table["possible_transition_count"].ge(
        result_table["observed_transition_count"]
    ).all()

    assert result_table["transition_density"].between(0.0, 1.0).all()