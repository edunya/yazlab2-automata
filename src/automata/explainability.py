"""
Explainability utilities for probabilistic automata decisions.

This module combines:
- seen/unseen symbolic state tracking
- Levenshtein mapping records
- transition probabilities
- path probability
- anomaly score
- threshold-based decision
- relative confidence score
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.automata.levenshtein import (
    LevenshteinStateMapper,
    StateMapping
)
from src.automata.probabilistic_automata import (
    PathScore,
    ProbabilisticAutomata,
    TransitionScore
)


def compute_relative_threshold_confidence(
    anomaly_score: float,
    threshold: float,
    epsilon: float = 1e-12
) -> float:
    """
    Compute a bounded decision-margin confidence score.

    Formula
    -------
    abs(anomaly_score - threshold) /
    (abs(anomaly_score) + abs(threshold) + epsilon)

    Interpretation
    --------------
    - Near 0: decision is close to the calibrated threshold.
    - Near 1: decision is far from the calibrated threshold.

    Important
    ---------
    This value is not a probability.
    """
    if not isinstance(anomaly_score, Real) or not isinstance(threshold, Real):
        raise TypeError("anomaly_score and threshold must be numerical.")

    anomaly_score = float(anomaly_score)
    threshold = float(threshold)

    if not np.isfinite(anomaly_score) or not np.isfinite(threshold):
        raise ValueError("anomaly_score and threshold must be finite.")

    if anomaly_score < 0 or threshold < 0:
        raise ValueError(
            "anomaly_score and threshold must be non-negative."
        )

    denominator = abs(anomaly_score) + abs(threshold) + epsilon

    if np.isclose(denominator, epsilon):
        return 0.0

    confidence = abs(anomaly_score - threshold) / denominator

    return float(np.clip(confidence, 0.0, 1.0))


@dataclass(frozen=True)
class AutomataDecisionExplanation:
    """
    Stores one explainable automata decision.
    """

    sequence_id: Optional[str]
    state_mappings: tuple[StateMapping, ...]
    transition_scores: tuple[TransitionScore, ...]
    path_probability: float
    log_path_probability: float
    anomaly_score: float
    threshold: float
    decision: int
    decision_label: str
    decision_margin: float
    confidence_score: float
    confidence_definition: str = "relative_threshold_margin_not_probability"

    @property
    def unseen_state_count(self) -> int:
        """
        Return number of unseen observed states.
        """
        return sum(
            1
            for mapping in self.state_mappings
            if mapping.is_unseen
        )

    @property
    def original_states(self) -> tuple[str, ...]:
        """
        Return originally observed symbolic states.
        """
        return tuple(
            mapping.original_state
            for mapping in self.state_mappings
        )

    @property
    def mapped_states(self) -> tuple[str, ...]:
        """
        Return mapped automata states.
        """
        return tuple(
            mapping.mapped_state
            for mapping in self.state_mappings
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-compatible explanation output.
        """
        state_trace = []

        for position, mapping in enumerate(self.state_mappings):
            state_record = mapping.as_dict()
            state_record["position"] = position
            state_trace.append(state_record)

        transition_trace = [
            {
                "transition_index": transition.transition_index,
                "from_state": transition.from_state,
                "to_state": transition.to_state,
                "probability": transition.probability,
                "log_probability": transition.log_probability,
                "anomaly_score": transition.anomaly_score
            }
            for transition in self.transition_scores
        ]

        return {
            "sequence_id": self.sequence_id,
            "original_states": list(self.original_states),
            "mapped_states": list(self.mapped_states),
            "state_trace": state_trace,
            "transition_trace": transition_trace,
            "unseen_state_count": self.unseen_state_count,
            "path_probability": self.path_probability,
            "log_path_probability": self.log_path_probability,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "decision": self.decision,
            "decision_label": self.decision_label,
            "decision_margin": self.decision_margin,
            "confidence_score": self.confidence_score,
            "confidence_definition": self.confidence_definition
        }


def explain_automata_decision(
    automata: ProbabilisticAutomata,
    mapper: LevenshteinStateMapper,
    observed_states: Sequence[str],
    threshold: float,
    sequence_id: Optional[str] = None
) -> AutomataDecisionExplanation:
    """
    Produce one explainable automata decision for an observed state path.

    Processing steps
    ----------------
    1. Map unseen states using Levenshtein nearest-state mapping.
    2. Score the mapped state path using fitted automata.
    3. Compare anomaly score with calibrated threshold.
    4. Produce JSON-compatible explanation details.
    """
    if not automata.is_fitted_:
        raise RuntimeError("ProbabilisticAutomata must be fitted first.")

    if set(mapper.known_states) != set(automata.states_):
        raise ValueError(
            "Mapper known states must match fitted automata states."
        )

    if not isinstance(threshold, Real):
        raise TypeError("threshold must be numerical.")

    threshold = float(threshold)

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")

    mapped_states, mappings = mapper.map_sequence(observed_states)

    path_score: PathScore = automata.score_path(
        states=mapped_states,
        sequence_id=sequence_id
    )

    anomaly_score = path_score.mean_negative_log_probability
    decision = int(anomaly_score >= threshold)
    decision_label = "anomaly" if decision == 1 else "normal"
    decision_margin = float(anomaly_score - threshold)

    confidence_score = compute_relative_threshold_confidence(
        anomaly_score=anomaly_score,
        threshold=threshold
    )

    return AutomataDecisionExplanation(
        sequence_id=sequence_id,
        state_mappings=mappings,
        transition_scores=path_score.transitions,
        path_probability=path_score.path_probability,
        log_path_probability=path_score.log_path_probability,
        anomaly_score=anomaly_score,
        threshold=threshold,
        decision=decision,
        decision_label=decision_label,
        decision_margin=decision_margin,
        confidence_score=confidence_score
    )


def explain_multiple_decisions(
    automata: ProbabilisticAutomata,
    mapper: LevenshteinStateMapper,
    observed_sequences: Mapping[str, Sequence[str]],
    threshold: float
) -> Dict[str, AutomataDecisionExplanation]:
    """
    Explain multiple independent observed symbolic paths.
    """
    if not observed_sequences:
        raise ValueError("observed_sequences must not be empty.")

    return {
        sequence_id: explain_automata_decision(
            automata=automata,
            mapper=mapper,
            observed_states=states,
            threshold=threshold,
            sequence_id=sequence_id
        )
        for sequence_id, states in observed_sequences.items()
    }