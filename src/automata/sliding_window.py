"""
Sliding window pattern extraction for symbolic automata sequences.

This module converts SAX symbol sequences into fixed-length patterns.
Each pattern will later represent an automata state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import numpy as np


@dataclass
class PatternWindowData:
    """
    Container for symbolic sliding-window patterns.

    Attributes
    ----------
    patterns:
        Pattern strings such as "abc" or "aabc".

    start_indices:
        Start positions inside the originating symbolic sequence.

    end_indices:
        Inclusive end positions inside the originating symbolic sequence.

    sequence_ids:
        Optional originating sequence identifiers.
        SKAB will use source_file values here.
    """

    patterns: np.ndarray
    start_indices: np.ndarray
    end_indices: np.ndarray
    sequence_ids: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.patterns))


def _validate_window_size(window_size: int) -> int:
    """
    Validate symbolic window size.
    """
    if isinstance(window_size, bool) or not isinstance(window_size, Integral):
        raise TypeError("window_size must be an integer.")

    window_size = int(window_size)

    if window_size <= 0:
        raise ValueError("window_size must be greater than zero.")

    return window_size


def _normalize_symbols(symbols: Sequence[str] | np.ndarray) -> np.ndarray:
    """
    Validate and normalize one SAX symbol sequence.
    """
    normalized = np.asarray(list(symbols), dtype=str)

    if normalized.ndim != 1:
        raise ValueError("symbols must be one-dimensional.")

    if normalized.size == 0:
        raise ValueError("symbols must not be empty.")

    invalid_symbols = [
        symbol
        for symbol in normalized.tolist()
        if len(symbol) != 1
    ]

    if invalid_symbols:
        raise ValueError(
            "Each SAX symbol must contain exactly one character."
        )

    return normalized


def create_symbol_patterns(
    symbols: Sequence[str] | np.ndarray,
    window_size: int,
    sequence_id: Optional[str] = None
) -> PatternWindowData:
    """
    Create fixed-length patterns from one SAX symbol sequence.

    Example
    -------
    symbols = [a, b, c, a], window_size = 3
    patterns = [abc, bca]
    """
    normalized_symbols = _normalize_symbols(symbols)
    window_size = _validate_window_size(window_size)

    if normalized_symbols.size < window_size:
        raise ValueError(
            "Symbol sequence is shorter than the requested window_size."
        )

    pattern_count = normalized_symbols.size - window_size + 1

    patterns = []
    start_indices = []
    end_indices = []

    for start_index in range(pattern_count):
        end_index = start_index + window_size - 1
        pattern = "".join(
            normalized_symbols[start_index:start_index + window_size].tolist()
        )

        patterns.append(pattern)
        start_indices.append(start_index)
        end_indices.append(end_index)

    sequence_ids = (
        np.full(pattern_count, sequence_id, dtype=object)
        if sequence_id is not None
        else None
    )

    return PatternWindowData(
        patterns=np.asarray(patterns, dtype=str),
        start_indices=np.asarray(start_indices, dtype=np.int64),
        end_indices=np.asarray(end_indices, dtype=np.int64),
        sequence_ids=sequence_ids
    )


def create_grouped_symbol_patterns(
    symbol_sequences: Mapping[str, Sequence[str] | np.ndarray],
    window_size: int
) -> PatternWindowData:
    """
    Create symbolic patterns independently for multiple sequences.

    This method prevents patterns from crossing sequence boundaries.
    For SKAB, each key should be one source_file value.
    """
    window_size = _validate_window_size(window_size)

    if not symbol_sequences:
        raise ValueError("symbol_sequences must not be empty.")

    pattern_parts = []
    start_parts = []
    end_parts = []
    sequence_id_parts = []

    for sequence_id, symbols in symbol_sequences.items():
        normalized_symbols = _normalize_symbols(symbols)

        if normalized_symbols.size < window_size:
            continue

        part = create_symbol_patterns(
            symbols=normalized_symbols,
            window_size=window_size,
            sequence_id=sequence_id
        )

        pattern_parts.append(part.patterns)
        start_parts.append(part.start_indices)
        end_parts.append(part.end_indices)
        sequence_id_parts.append(part.sequence_ids)

    if not pattern_parts:
        raise ValueError(
            "No symbolic patterns could be created from the provided sequences."
        )

    return PatternWindowData(
        patterns=np.concatenate(pattern_parts),
        start_indices=np.concatenate(start_parts),
        end_indices=np.concatenate(end_parts),
        sequence_ids=np.concatenate(sequence_id_parts)
    )