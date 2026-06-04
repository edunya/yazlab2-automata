"""
Tests for Piecewise Aggregate Approximation.
"""

import numpy as np
import pytest

from src.automata.paa import PAATransformer, paa_transform


def test_paa_transform_with_even_segments():
    series = np.asarray([1.0, 2.0, 3.0, 4.0])

    transformed = paa_transform(series, n_segments=2)

    assert np.allclose(transformed, [1.5, 3.5])


def test_paa_transform_handles_non_divisible_length():
    series = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    transformed = paa_transform(series, n_segments=2)

    assert np.allclose(transformed, [1.8, 4.2])


def test_paa_transformer_wrapper():
    transformer = PAATransformer(n_segments=3)

    transformed = transformer.transform(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )

    assert np.allclose(transformed, [1.5, 3.5, 5.5])


def test_paa_identity_when_segment_count_matches_series_length():
    series = np.asarray([4.0, 3.0, 2.0, 1.0])

    transformed = paa_transform(series, n_segments=4)

    assert np.array_equal(transformed, series)
    assert transformed is not series


@pytest.mark.parametrize("n_segments", [0, -1, 5])
def test_paa_rejects_invalid_segment_count(n_segments: int):
    series = np.asarray([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        paa_transform(series, n_segments=n_segments)