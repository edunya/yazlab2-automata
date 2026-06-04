"""
Tests for explainable probabilistic automata decisions.
"""

import json

import pytest

from src.automata.explainability import (
    compute_relative_threshold_confidence,
    explain_automata_decision,
    explain_multiple_decisions
)
from src.automata.levenshtein import LevenshteinStateMapper
from src.automata.probabilistic_automata import ProbabilisticAutomata


def build_example_automata() -> tuple[
    ProbabilisticAutomata,
    LevenshteinStateMapper
]:
    """
    Build automata with common aa<->ab transitions and a known cc state.
    """
    automata = ProbabilisticAutomata(smoothing=1e-3)

    automata.fit({
        "common_run": [
            "aa", "ab", "aa", "ab", "aa",
            "ab", "aa", "ab", "aa"
        ],
        "other_known_run": ["cc", "cc"]
    })

    mapper = LevenshteinStateMapper(automata.states_)

    return automata, mapper


def test_relative_threshold_confidence_is_bounded_and_not_probability():
    confidence_near = compute_relative_threshold_confidence(
        anomaly_score=1.01,
        threshold=1.0
    )

    confidence_far = compute_relative_threshold_confidence(
        anomaly_score=4.0,
        threshold=1.0
    )

    assert 0.0 <= confidence_near <= 1.0
    assert 0.0 <= confidence_far <= 1.0
    assert confidence_far > confidence_near


def test_explanation_for_seen_normal_path():
    automata, mapper = build_example_automata()

    explanation = explain_automata_decision(
        automata=automata,
        mapper=mapper,
        observed_states=["aa", "ab", "aa"],
        threshold=1.0,
        sequence_id="seen_normal_path"
    )

    assert explanation.sequence_id == "seen_normal_path"
    assert explanation.original_states == ("aa", "ab", "aa")
    assert explanation.mapped_states == ("aa", "ab", "aa")
    assert explanation.unseen_state_count == 0
    assert explanation.decision == 0
    assert explanation.decision_label == "normal"
    assert explanation.anomaly_score < explanation.threshold


def test_explanation_maps_unseen_state_before_scoring():
    automata, mapper = build_example_automata()

    explanation = explain_automata_decision(
        automata=automata,
        mapper=mapper,
        observed_states=["aa", "bb", "aa"],
        threshold=1.0,
        sequence_id="unseen_path"
    )

    assert explanation.original_states == ("aa", "bb", "aa")
    assert explanation.mapped_states == ("aa", "ab", "aa")
    assert explanation.unseen_state_count == 1

    unseen_record = explanation.state_mappings[1]

    assert unseen_record.original_state == "bb"
    assert unseen_record.mapped_state == "ab"
    assert unseen_record.distance == 1
    assert unseen_record.status == "unseen"


def test_rare_seen_transition_can_be_explained_as_anomaly():
    automata, mapper = build_example_automata()

    explanation = explain_automata_decision(
        automata=automata,
        mapper=mapper,
        observed_states=["aa", "cc"],
        threshold=1.0,
        sequence_id="rare_transition_path"
    )

    assert explanation.unseen_state_count == 0
    assert explanation.decision == 1
    assert explanation.decision_label == "anomaly"
    assert explanation.anomaly_score >= explanation.threshold
    assert explanation.path_probability > 0


def test_explanation_dictionary_is_json_serializable_and_complete():
    automata, mapper = build_example_automata()

    explanation = explain_automata_decision(
        automata=automata,
        mapper=mapper,
        observed_states=["aa", "bb", "aa"],
        threshold=1.0,
        sequence_id="json_path"
    )

    result = explanation.as_dict()

    json_text = json.dumps(result)

    assert isinstance(json_text, str)
    assert result["sequence_id"] == "json_path"
    assert result["decision"] in {0, 1}
    assert "state_trace" in result
    assert "transition_trace" in result
    assert "path_probability" in result
    assert "log_path_probability" in result
    assert "anomaly_score" in result
    assert "threshold" in result
    assert "confidence_score" in result
    assert result["confidence_definition"] == (
        "relative_threshold_margin_not_probability"
    )


def test_explain_multiple_decisions_keeps_sequence_ids():
    automata, mapper = build_example_automata()

    explanations = explain_multiple_decisions(
        automata=automata,
        mapper=mapper,
        observed_sequences={
            "path_1": ["aa", "ab", "aa"],
            "path_2": ["aa", "cc"]
        },
        threshold=1.0
    )

    assert set(explanations.keys()) == {"path_1", "path_2"}
    assert explanations["path_1"].sequence_id == "path_1"
    assert explanations["path_2"].sequence_id == "path_2"


def test_mapper_must_match_automata_vocabulary():
    automata, _ = build_example_automata()

    incorrect_mapper = LevenshteinStateMapper(["aa", "ab"])

    with pytest.raises(ValueError):
        explain_automata_decision(
            automata=automata,
            mapper=incorrect_mapper,
            observed_states=["aa", "ab"],
            threshold=1.0
        )