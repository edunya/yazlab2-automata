"""
Probabilistic automata core.

This module learns symbolic state transitions from uninterrupted normal
training runs and scores observed symbolic paths using transition
probabilities.

Important rules
---------------
- The automata is fitted only on normal training runs.
- Normal runs separated by an anomaly are never concatenated.
- Unseen states are not silently handled in this module.
  Levenshtein-based mapping will be implemented separately.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class UnseenStateError(ValueError):
    """
    Raised when a symbolic state is absent from the fitted automata vocabulary.
    """


@dataclass(frozen=True)
class TransitionScore:
    """
    Score information for one symbolic state transition.
    """

    transition_index: int
    from_state: str
    to_state: str
    probability: float
    log_probability: float
    anomaly_score: float


@dataclass(frozen=True)
class PathScore:
    """
    Score information for one symbolic state path.
    """

    sequence_id: Optional[str]
    states: tuple[str, ...]
    transitions: tuple[TransitionScore, ...]
    path_probability: float
    log_path_probability: float
    mean_negative_log_probability: float

    @property
    def transition_count(self) -> int:
        """
        Return number of transitions inside the scored path.
        """
        return len(self.transitions)


def validate_state_sequence(
    states: Sequence[str],
    sequence_name: str = "states"
) -> list[str]:
    """
    Validate one symbolic state sequence.

    Parameters
    ----------
    states:
        Sequence of pattern/state strings such as ["aab", "abb", "bbc"].

    sequence_name:
        Name used in validation error messages.

    Returns
    -------
    list[str]
        Validated symbolic states.
    """
    if isinstance(states, str):
        raise TypeError(
            f"{sequence_name} must be a sequence of states, not one string."
        )

    validated_states = list(states)

    if not validated_states:
        raise ValueError(f"{sequence_name} must not be empty.")

    invalid_states = [
        state
        for state in validated_states
        if not isinstance(state, str) or not state.strip()
    ]

    if invalid_states:
        raise ValueError(
            f"{sequence_name} contains invalid symbolic states."
        )

    return validated_states


def validate_binary_labels(
    labels: Sequence[int] | np.ndarray,
    expected_length: int,
    sequence_name: str
) -> np.ndarray:
    """
    Validate binary pattern-level labels.
    """
    label_values = np.asarray(labels)

    if label_values.ndim != 1:
        raise ValueError(f"Labels for {sequence_name} must be one-dimensional.")

    if len(label_values) != expected_length:
        raise ValueError(
            f"State and label counts differ for {sequence_name}: "
            f"{expected_length} states versus {len(label_values)} labels."
        )

    unique_labels = set(label_values.tolist())

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Labels for {sequence_name} must contain only 0 and 1."
        )

    return label_values.astype(int)


def extract_contiguous_normal_runs(
    pattern_sequences: Mapping[str, Sequence[str]],
    label_sequences: Mapping[str, Sequence[int] | np.ndarray]
) -> Dict[str, list[str]]:
    """
    Extract uninterrupted normal symbolic runs from labelled training paths.

    Each state label is expected to represent the anomaly status associated
    with that symbolic pattern. A label of 0 means normal and a label of 1
    means anomaly.

    Crucial behavior
    ----------------
    When anomaly patterns separate two normal regions, the two normal regions
    become different runs. Therefore, no artificial transition is introduced
    across the removed anomaly region.

    Example
    -------
    states = [aa, ab, zz, bc, cc]
    labels = [0,  0,  1,  0,  0]

    Returned normal runs:
    [aa, ab] and [bc, cc]

    No artificial transition ab -> bc is created.
    """
    if not pattern_sequences:
        raise ValueError("pattern_sequences must not be empty.")

    if set(pattern_sequences.keys()) != set(label_sequences.keys()):
        raise ValueError(
            "pattern_sequences and label_sequences must have identical keys."
        )

    normal_runs: Dict[str, list[str]] = {}

    for sequence_id, raw_states in pattern_sequences.items():
        states = validate_state_sequence(
            raw_states,
            sequence_name=f"states[{sequence_id}]"
        )
        labels = validate_binary_labels(
            labels=label_sequences[sequence_id],
            expected_length=len(states),
            sequence_name=sequence_id
        )

        current_run: list[str] = []
        run_number = 0

        def save_current_run() -> None:
            nonlocal current_run, run_number

            if current_run:
                run_id = f"{sequence_id}::normal_run_{run_number}"
                normal_runs[run_id] = current_run.copy()
                run_number += 1
                current_run = []

        for state, label in zip(states, labels):
            if label == 0:
                current_run.append(state)
            else:
                save_current_run()

        save_current_run()

    if not normal_runs:
        raise ValueError("No normal symbolic runs were found in training data.")

    return normal_runs


@dataclass
class ProbabilisticAutomata:
    """
    First-order probabilistic automata over symbolic pattern states.

    Transition probabilities use additive smoothing:

        P(next | current) =
        (transition_count + smoothing) /
        (all_outgoing_count + smoothing * state_count)
    """

    smoothing: float = 1e-8
    states_: tuple[str, ...] = field(default_factory=tuple, init=False)
    state_counts_: Counter[str] = field(default_factory=Counter, init=False)
    transition_counts_: Dict[str, Counter[str]] = field(
        default_factory=dict,
        init=False
    )
    outgoing_counts_: Counter[str] = field(
        default_factory=Counter,
        init=False
    )
    training_sequence_count_: int = field(default=0, init=False)
    training_transition_count_: int = field(default=0, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.smoothing, Real):
            raise TypeError("smoothing must be numerical.")

        if self.smoothing <= 0:
            raise ValueError("smoothing must be greater than zero.")

        self.smoothing = float(self.smoothing)

    def fit(
        self,
        normal_sequences: Mapping[str, Sequence[str]]
    ) -> "ProbabilisticAutomata":
        """
        Fit transition probabilities from uninterrupted normal sequences only.
        """
        if not normal_sequences:
            raise ValueError("normal_sequences must not be empty.")

        state_counts: Counter[str] = Counter()
        outgoing_counts: Counter[str] = Counter()
        transition_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

        transition_total = 0

        for sequence_id, raw_states in normal_sequences.items():
            states = validate_state_sequence(
                raw_states,
                sequence_name=f"normal_sequences[{sequence_id}]"
            )

            state_counts.update(states)

            for from_state, to_state in zip(states[:-1], states[1:]):
                transition_counts[from_state][to_state] += 1
                outgoing_counts[from_state] += 1
                transition_total += 1

        if transition_total == 0:
            raise ValueError(
                "At least one normal state transition is required "
                "to fit probabilistic automata."
            )

        self.states_ = tuple(sorted(state_counts.keys()))
        self.state_counts_ = state_counts
        self.outgoing_counts_ = outgoing_counts
        self.transition_counts_ = {
            state: Counter(counts)
            for state, counts in transition_counts.items()
        }
        self.training_sequence_count_ = len(normal_sequences)
        self.training_transition_count_ = transition_total
        self.is_fitted_ = True

        return self

    def fit_from_labeled_sequences(
        self,
        pattern_sequences: Mapping[str, Sequence[str]],
        label_sequences: Mapping[str, Sequence[int] | np.ndarray]
    ) -> "ProbabilisticAutomata":
        """
        Extract normal runs from labelled training sequences and fit model.
        """
        normal_runs = extract_contiguous_normal_runs(
            pattern_sequences=pattern_sequences,
            label_sequences=label_sequences
        )

        return self.fit(normal_runs)

    def _require_fitted(self) -> None:
        """
        Raise if automata has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("ProbabilisticAutomata must be fitted first.")

    def _validate_seen_states(self, states: Sequence[str]) -> list[str]:
        """
        Validate states and reject vocabulary states unseen during training.
        """
        self._require_fitted()

        validated_states = validate_state_sequence(states)

        known_states = set(self.states_)
        unseen_states = sorted(set(validated_states).difference(known_states))

        if unseen_states:
            raise UnseenStateError(
                "Unseen symbolic states detected: "
                f"{unseen_states}. "
                "Apply Levenshtein mapping before automata scoring."
            )

        return validated_states

    def transition_probability(
        self,
        from_state: str,
        to_state: str
    ) -> float:
        """
        Return smoothed transition probability P(to_state | from_state).
        """
        self._validate_seen_states([from_state, to_state])

        state_count = len(self.states_)

        observed_transition_count = self.transition_counts_.get(
            from_state,
            Counter()
        ).get(to_state, 0)

        outgoing_count = self.outgoing_counts_.get(from_state, 0)

        numerator = observed_transition_count + self.smoothing
        denominator = outgoing_count + self.smoothing * state_count

        return float(numerator / denominator)

    def transition_matrix(self) -> pd.DataFrame:
        """
        Return a row-normalized transition probability matrix.

        This matrix will later be used for the transition heatmap.
        """
        self._require_fitted()

        matrix = np.empty(
            (len(self.states_), len(self.states_)),
            dtype=np.float64
        )

        for row_index, from_state in enumerate(self.states_):
            for column_index, to_state in enumerate(self.states_):
                matrix[row_index, column_index] = self.transition_probability(
                    from_state=from_state,
                    to_state=to_state
                )

        return pd.DataFrame(
            matrix,
            index=list(self.states_),
            columns=list(self.states_)
        )

    def score_path(
        self,
        states: Sequence[str],
        sequence_id: Optional[str] = None
    ) -> PathScore:
        """
        Score one symbolic state path.

        The final anomaly score is the mean negative log transition
        probability. Larger values indicate less expected behavior.
        """
        validated_states = self._validate_seen_states(states)

        if len(validated_states) < 2:
            raise ValueError(
                "At least two states are required to score a path."
            )

        transition_scores = []
        log_path_probability = 0.0

        for transition_index, (from_state, to_state) in enumerate(
            zip(validated_states[:-1], validated_states[1:])
        ):
            probability = self.transition_probability(from_state, to_state)
            log_probability = float(np.log(probability))
            anomaly_score = float(-log_probability)

            transition_scores.append(
                TransitionScore(
                    transition_index=transition_index,
                    from_state=from_state,
                    to_state=to_state,
                    probability=probability,
                    log_probability=log_probability,
                    anomaly_score=anomaly_score
                )
            )

            log_path_probability += log_probability

        mean_negative_log_probability = float(
            -log_path_probability / len(transition_scores)
        )

        path_probability = float(np.exp(log_path_probability))

        return PathScore(
            sequence_id=sequence_id,
            states=tuple(validated_states),
            transitions=tuple(transition_scores),
            path_probability=path_probability,
            log_path_probability=float(log_path_probability),
            mean_negative_log_probability=mean_negative_log_probability
        )

    def score_paths(
        self,
        pattern_sequences: Mapping[str, Sequence[str]]
    ) -> Dict[str, PathScore]:
        """
        Score multiple symbolic state paths independently.
        """
        if not pattern_sequences:
            raise ValueError("pattern_sequences must not be empty.")

        return {
            sequence_id: self.score_path(
                states=states,
                sequence_id=sequence_id
            )
            for sequence_id, states in pattern_sequences.items()
        }

    def summary(self) -> Dict[str, Any]:
        """
        Return fitted automata information for experiment logging.
        """
        self._require_fitted()

        return {
            "model": "probabilistic_automata",
            "state_count": len(self.states_),
            "states": list(self.states_),
            "training_sequence_count": self.training_sequence_count_,
            "training_transition_count": self.training_transition_count_,
            "smoothing": self.smoothing,
            "learning_strategy": "normal_train_runs",
            "score_strategy": "mean_negative_log_probability"
        }