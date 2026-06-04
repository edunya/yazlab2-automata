"""
Tests for probabilistic automata state transition learning and scoring.
"""

import numpy as np
import pytest

from src.automata.probabilistic_automata import (
    ProbabilisticAutomata,
    UnseenStateError,
    extract_contiguous_normal_runs
)


def test_extract_normal_runs_does_not_bridge_anomaly_regions():
    pattern_sequences = {
        "file_a": ["aa", "ab", "zz", "bc", "cc"]
    }
    label_sequences = {
        "file_a": [0, 0, 1, 0, 0]
    }

    normal_runs = extract_contiguous_normal_runs(
        pattern_sequences=pattern_sequences,
        label_sequences=label_sequences
    )

    assert list(normal_runs.values()) == [
        ["aa", "ab"],
        ["bc", "cc"]
    ]

    transitions = [
        (states[index], states[index + 1])
        for states in normal_runs.values()
        for index in range(len(states) - 1)
    ]

    assert ("ab", "bc") not in transitions


def test_automata_learns_smoothed_transition_probabilities():
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit({
        "run_1": ["aa", "ab", "aa"],
        "run_2": ["aa", "ab"]
    })

    assert automata.states_ == ("aa", "ab")
    assert automata.training_sequence_count_ == 2
    assert automata.training_transition_count_ == 3

    assert automata.transition_probability("aa", "ab") == pytest.approx(0.75)
    assert automata.transition_probability("aa", "aa") == pytest.approx(0.25)
    assert automata.transition_probability("ab", "aa") == pytest.approx(2 / 3)


def test_transition_matrix_rows_are_probabilities():
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit({
        "run_1": ["aa", "ab", "aa"],
        "run_2": ["aa", "ab"]
    })

    matrix = automata.transition_matrix()

    assert matrix.shape == (2, 2)
    assert matrix.index.tolist() == ["aa", "ab"]
    assert matrix.columns.tolist() == ["aa", "ab"]
    assert np.allclose(matrix.sum(axis=1).to_numpy(), [1.0, 1.0])


def test_path_score_returns_probability_and_mean_negative_log_probability():
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit({
        "run_1": ["aa", "ab", "aa"],
        "run_2": ["aa", "ab"]
    })

    score = automata.score_path(
        states=["aa", "ab", "aa"],
        sequence_id="validation_run"
    )

    expected_probability = 0.75 * (2 / 3)
    expected_mean_negative_log = -np.log(expected_probability) / 2

    assert score.sequence_id == "validation_run"
    assert score.transition_count == 2
    assert score.path_probability == pytest.approx(expected_probability)
    assert score.log_path_probability == pytest.approx(
        np.log(expected_probability)
    )
    assert score.mean_negative_log_probability == pytest.approx(
        expected_mean_negative_log
    )


def test_fit_from_labeled_sequences_uses_only_normal_states_and_transitions():
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit_from_labeled_sequences(
        pattern_sequences={
            "file_a": ["aa", "ab", "zz", "bc", "cc"]
        },
        label_sequences={
            "file_a": [0, 0, 1, 0, 0]
        }
    )

    assert set(automata.states_) == {"aa", "ab", "bc", "cc"}
    assert "zz" not in automata.states_
    assert automata.training_transition_count_ == 2


def test_automata_rejects_unseen_states_before_levenshtein_mapping():
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit({
        "run_1": ["aa", "ab", "aa"]
    })

    with pytest.raises(UnseenStateError):
        automata.score_path(["aa", "zz"])


def test_automata_requires_at_least_one_transition_for_training():
    automata = ProbabilisticAutomata()

    with pytest.raises(ValueError):
        automata.fit({
            "single_normal_state": ["aa"]
        })