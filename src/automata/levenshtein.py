"""
Levenshtein-based unseen symbolic state mapping.

This module maps symbolic patterns that were not observed during automata
training to the nearest known state using edit distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


def levenshtein_distance(source: str, target: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Supported operations:
    - insertion
    - deletion
    - substitution
    """
    if not isinstance(source, str) or not isinstance(target, str):
        raise TypeError("source and target must be strings.")

    if source == target:
        return 0

    if len(source) == 0:
        return len(target)

    if len(target) == 0:
        return len(source)

    previous_row = list(range(len(target) + 1))

    for source_index, source_character in enumerate(source, start=1):
        current_row = [source_index]

        for target_index, target_character in enumerate(target, start=1):
            insertion_cost = current_row[target_index - 1] + 1
            deletion_cost = previous_row[target_index] + 1
            substitution_cost = previous_row[target_index - 1] + (
                0 if source_character == target_character else 1
            )

            current_row.append(
                min(insertion_cost, deletion_cost, substitution_cost)
            )

        previous_row = current_row

    return previous_row[-1]


def validate_symbolic_state(state: str, state_name: str = "state") -> str:
    """
    Validate one symbolic state/pattern string.
    """
    if not isinstance(state, str):
        raise TypeError(f"{state_name} must be a string.")

    if not state:
        raise ValueError(f"{state_name} must not be empty.")

    return state


@dataclass(frozen=True)
class StateMapping:
    """
    Stores how one observed symbolic state was mapped.

    Attributes
    ----------
    original_state:
        State observed in validation/test data.

    mapped_state:
        Known training state used by the automata.

    distance:
        Levenshtein distance between original and mapped state.

    status:
        Either "seen" or "unseen".

    mapping_method:
        Method used to obtain the mapped state.
    """

    original_state: str
    mapped_state: str
    distance: int
    status: str
    mapping_method: str

    @property
    def is_unseen(self) -> bool:
        """
        Return True when the original state was absent during training.
        """
        return self.status == "unseen"

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-compatible representation.
        """
        return {
            "original_state": self.original_state,
            "mapped_state": self.mapped_state,
            "distance": self.distance,
            "status": self.status,
            "mapping_method": self.mapping_method
        }


class LevenshteinStateMapper:
    """
    Maps unseen symbolic states to nearest known automata states.

    Tie-breaking
    ------------
    When multiple known states have the same minimum distance,
    the alphabetically first state is selected. This keeps mapping
    deterministic and reproducible.
    """

    def __init__(self, known_states: Sequence[str]) -> None:
        if isinstance(known_states, str):
            raise TypeError("known_states must be a sequence of states.")

        validated_states = [
            validate_symbolic_state(state, state_name="known_state")
            for state in known_states
        ]

        if not validated_states:
            raise ValueError("known_states must not be empty.")

        self.known_states = tuple(sorted(set(validated_states)))
        self.known_state_set = set(self.known_states)

    def map_state(self, observed_state: str) -> StateMapping:
        """
        Map one observed state to a known automata state.
        """
        observed_state = validate_symbolic_state(
            observed_state,
            state_name="observed_state"
        )

        if observed_state in self.known_state_set:
            return StateMapping(
                original_state=observed_state,
                mapped_state=observed_state,
                distance=0,
                status="seen",
                mapping_method="exact_match"
            )

        candidates = [
            (
                levenshtein_distance(observed_state, known_state),
                known_state
            )
            for known_state in self.known_states
        ]

        minimum_distance, selected_state = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1])
        )

        return StateMapping(
            original_state=observed_state,
            mapped_state=selected_state,
            distance=int(minimum_distance),
            status="unseen",
            mapping_method="levenshtein_nearest_state"
        )

    def map_sequence(
        self,
        observed_states: Sequence[str]
    ) -> tuple[tuple[str, ...], tuple[StateMapping, ...]]:
        """
        Map all states in one observed symbolic path.

        Returns
        -------
        mapped_states:
            State sequence that can be scored by fitted automata.

        mappings:
            Detailed seen/unseen mapping records.
        """
        if isinstance(observed_states, str):
            raise TypeError(
                "observed_states must be a sequence of states, "
                "not one string."
            )

        if not observed_states:
            raise ValueError("observed_states must not be empty.")

        mappings = tuple(
            self.map_state(observed_state)
            for observed_state in observed_states
        )

        mapped_states = tuple(
            mapping.mapped_state
            for mapping in mappings
        )

        return mapped_states, mappings

    def summary(self) -> Dict[str, Any]:
        """
        Return mapper settings for experiment logging.
        """
        return {
            "known_state_count": len(self.known_states),
            "known_states": list(self.known_states),
            "unseen_mapping_strategy": "levenshtein_nearest_state",
            "unseen_tie_break": "alphabetical_state"
        }