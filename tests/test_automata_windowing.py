"""
Tests for symbolic sliding-window pattern extraction.
"""

import numpy as np
import pytest

from src.automata.paa import PAATransformer
from src.automata.sax import SAXDiscretizer
from src.automata.sliding_window import (
    create_grouped_symbol_patterns,
    create_symbol_patterns
)


def test_create_symbol_patterns():
    symbols = np.asarray(["a", "b", "c", "a"])

    patterns = create_symbol_patterns(
        symbols=symbols,
        window_size=3
    )

    assert patterns.patterns.tolist() == ["abc", "bca"]
    assert patterns.start_indices.tolist() == [0, 1]
    assert patterns.end_indices.tolist() == [2, 3]
    assert patterns.sequence_ids is None


def test_grouped_patterns_do_not_cross_sequence_boundaries():
    sequences = {
        "file_a": np.asarray(["a", "b", "c"]),
        "file_b": np.asarray(["a", "a", "b"])
    }

    patterns = create_grouped_symbol_patterns(
        symbol_sequences=sequences,
        window_size=2
    )

    assert patterns.patterns.tolist() == ["ab", "bc", "aa", "ab"]
    assert "ca" not in patterns.patterns.tolist()
    assert patterns.sequence_ids.tolist() == [
        "file_a", "file_a", "file_b", "file_b"
    ]


def test_symbol_pattern_generation_rejects_too_short_sequence():
    symbols = np.asarray(["a", "b"])

    with pytest.raises(ValueError):
        create_symbol_patterns(
            symbols=symbols,
            window_size=3
        )


def test_paa_sax_and_sliding_window_can_be_chained():
    pc1_series = np.asarray(
        [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
    )

    paa_values = PAATransformer(n_segments=8).transform(pc1_series)

    discretizer = SAXDiscretizer(alphabet_size=3)
    symbols = discretizer.fit_transform(paa_values)

    patterns = create_symbol_patterns(
        symbols=symbols,
        window_size=4
    )

    assert len(patterns) == 5
    assert all(len(pattern) == 4 for pattern in patterns.patterns.tolist())