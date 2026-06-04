"""
Tests for SAX symbolic discretization.
"""

import numpy as np
import pytest

from src.automata.sax import (
    SAXDiscretizer,
    gaussian_breakpoints,
    generate_symbols
)


def test_generate_symbols():
    symbols = generate_symbols(alphabet_size=4)

    assert symbols.tolist() == ["a", "b", "c", "d"]


def test_gaussian_breakpoints_for_alphabet_size_three():
    breakpoints = gaussian_breakpoints(alphabet_size=3)

    assert breakpoints.shape == (2,)
    assert np.allclose(
        breakpoints,
        [-0.4307273, 0.4307273],
        atol=1e-6
    )


def test_sax_fit_transform_maps_values_to_symbols():
    values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])

    discretizer = SAXDiscretizer(alphabet_size=3)
    symbols = discretizer.fit_transform(values)

    assert symbols.tolist() == ["a", "a", "b", "c", "c"]


def test_sax_transform_uses_training_statistics_only():
    train_values = np.asarray([-1.0, 0.0, 1.0])
    test_values = np.asarray([100.0, 101.0])

    discretizer = SAXDiscretizer(alphabet_size=3)
    discretizer.fit(train_values)

    test_symbols = discretizer.transform(test_values)

    assert test_symbols.tolist() == ["c", "c"]
    assert discretizer.mean_ == 0.0


def test_sax_constant_training_series_is_supported():
    train_values = np.asarray([5.0, 5.0, 5.0])

    discretizer = SAXDiscretizer(alphabet_size=3)
    symbols = discretizer.fit_transform(train_values)

    assert symbols.tolist() == ["b", "b", "b"]
    assert discretizer.constant_training_series_ is True
    assert discretizer.scale_ == 1.0


def test_sax_transform_requires_fit():
    discretizer = SAXDiscretizer(alphabet_size=3)

    with pytest.raises(RuntimeError):
        discretizer.transform(np.asarray([0.0, 1.0]))


@pytest.mark.parametrize("alphabet_size", [1, 27])
def test_sax_rejects_invalid_alphabet_size(alphabet_size: int):
    with pytest.raises(ValueError):
        SAXDiscretizer(alphabet_size=alphabet_size)