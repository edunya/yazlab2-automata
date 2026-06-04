"""
Tests for robustness and automata parameter-analysis utilities.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.automata.explainability import explain_automata_decision
from src.automata.levenshtein import LevenshteinStateMapper
from src.automata.probabilistic_automata import ProbabilisticAutomata
from src.data.splitting import DatasetSplit
from src.experiments.scenarios import (
    GaussianNoiseSetting,
    build_config_for_automata_setting,
    build_gaussian_noise_settings,
    create_automata_parameter_grid,
    inject_gaussian_noise_into_test_rows,
    summarize_unseen_automata_decisions
)
from src.utils.config_loader import load_config


def make_noise_test_data():
    X = pd.DataFrame({
        "sensor_a": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
        "sensor_b": [10.0, 12.0, 14.0, 16.0, 30.0, 31.0, 32.0, 33.0]
    })

    split = DatasetSplit(
        train_indices=np.asarray([0, 1, 2, 3]),
        validation_indices=np.asarray([4, 5]),
        test_indices=np.asarray([6, 7]),
        split_name="synthetic_noise_split"
    )

    return X, split


def test_gaussian_noise_changes_only_test_rows_and_is_reproducible():
    X, split = make_noise_test_data()

    setting = GaussianNoiseSetting(
        level=0.10,
        seed=2026
    )

    first_noisy_X = inject_gaussian_noise_into_test_rows(
        X=X,
        split=split,
        setting=setting
    )

    second_noisy_X = inject_gaussian_noise_into_test_rows(
        X=X,
        split=split,
        setting=setting
    )

    assert np.allclose(
        first_noisy_X.iloc[split.train_indices].to_numpy(),
        X.iloc[split.train_indices].to_numpy()
    )

    assert np.allclose(
        first_noisy_X.iloc[split.validation_indices].to_numpy(),
        X.iloc[split.validation_indices].to_numpy()
    )

    assert not np.allclose(
        first_noisy_X.iloc[split.test_indices].to_numpy(),
        X.iloc[split.test_indices].to_numpy()
    )

    assert np.allclose(
        first_noisy_X.to_numpy(),
        second_noisy_X.to_numpy()
    )


def test_gaussian_noise_scale_depends_only_on_training_features():
    X, split = make_noise_test_data()

    changed_test_X = X.copy()
    changed_test_X.iloc[split.test_indices] = (
        changed_test_X.iloc[split.test_indices] + 10000.0
    )

    setting = GaussianNoiseSetting(
        level=0.10,
        seed=2026
    )

    noisy_original = inject_gaussian_noise_into_test_rows(
        X=X,
        split=split,
        setting=setting
    )

    noisy_changed_test = inject_gaussian_noise_into_test_rows(
        X=changed_test_X,
        split=split,
        setting=setting
    )

    original_noise = (
        noisy_original.iloc[split.test_indices].to_numpy()
        - X.iloc[split.test_indices].to_numpy()
    )

    changed_test_noise = (
        noisy_changed_test.iloc[split.test_indices].to_numpy()
        - changed_test_X.iloc[split.test_indices].to_numpy()
    )

    assert np.allclose(original_noise, changed_test_noise)


def test_zero_noise_returns_unchanged_copy():
    X, split = make_noise_test_data()

    noisy_X = inject_gaussian_noise_into_test_rows(
        X=X,
        split=split,
        setting=GaussianNoiseSetting(level=0.0, seed=42)
    )

    assert np.allclose(noisy_X.to_numpy(), X.to_numpy())
    assert noisy_X is not X


def test_noise_setting_rejects_negative_level():
    X, split = make_noise_test_data()

    with pytest.raises(ValueError):
        inject_gaussian_noise_into_test_rows(
            X=X,
            split=split,
            setting=GaussianNoiseSetting(level=-0.1, seed=42)
        )


def test_gaussian_noise_settings_are_loaded_from_config():
    config = load_config()

    settings = build_gaussian_noise_settings(config)

    assert [setting.level for setting in settings] == [
        0.01, 0.05, 0.10
    ]
    assert all(setting.retrain_model is False for setting in settings)
    assert settings[0].scenario_name == "gaussian_noise_0.01"


def test_automata_parameter_grid_contains_sixteen_settings():
    config = load_config()

    settings = create_automata_parameter_grid(config)

    combinations = {
        (setting.window_size, setting.alphabet_size)
        for setting in settings
    }

    assert len(settings) == 16
    assert len(combinations) == 16
    assert (3, 3) in combinations
    assert (6, 6) in combinations


def test_automata_parameter_setting_does_not_modify_base_config():
    base_config = load_config()
    original_config = deepcopy(base_config)

    setting = create_automata_parameter_grid(base_config)[-1]

    adjusted_config = build_config_for_automata_setting(
        config=base_config,
        setting=setting
    )

    assert base_config == original_config
    assert adjusted_config["automata"]["default_window_size"] == 6
    assert adjusted_config["automata"]["default_alphabet_size"] == 6
    assert adjusted_config["automata"]["context_length"] == 32


def test_unseen_summary_separates_seen_and_unseen_involved_decisions():
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit({
        "normal_run": ["aa", "ab", "aa", "ab", "aa"],
        "known_other_state": ["cc", "cc"]
    })

    mapper = LevenshteinStateMapper(automata.states_)

    seen_explanation = explain_automata_decision(
        automata=automata,
        mapper=mapper,
        observed_states=["aa", "ab"],
        threshold=1.0,
        sequence_id="seen"
    )

    unseen_explanation = explain_automata_decision(
        automata=automata,
        mapper=mapper,
        observed_states=["aa", "bb"],
        threshold=1.0,
        sequence_id="unseen"
    )

    explanations = [seen_explanation, unseen_explanation]
    scores = [
        seen_explanation.anomaly_score,
        unseen_explanation.anomaly_score
    ]

    summary = summarize_unseen_automata_decisions(
        explanations=explanations,
        labels=[0, 1],
        scores=scores,
        threshold=1.0
    )

    assert summary.total_decisions == 2
    assert summary.seen_only_decisions == 1
    assert summary.unseen_involved_decisions == 1
    assert summary.unseen_state_occurrences == 1
    assert summary.unseen_decision_ratio == pytest.approx(0.5)
    assert summary.seen_only_metrics is not None
    assert summary.unseen_involved_metrics is not None