"""
Piecewise Aggregate Approximation (PAA).

PAA reduces a one-dimensional time series into a smaller number
of segment means. In this project, the input series is expected
to be the PC1 representation produced by the automata preprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


def validate_univariate_series(
    series: Sequence[float] | np.ndarray,
    series_name: str = "series"
) -> np.ndarray:
    """
    Convert input into a finite one-dimensional NumPy array.

    Parameters
    ----------
    series:
        One-dimensional numerical sequence.

    series_name:
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Validated float64 one-dimensional array.
    """
    values = np.asarray(series, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(f"{series_name} must be one-dimensional.")

    if values.size == 0:
        raise ValueError(f"{series_name} must not be empty.")

    if not np.isfinite(values).all():
        raise ValueError(f"{series_name} contains non-finite values.")

    return values


def paa_transform(
    series: Sequence[float] | np.ndarray,
    n_segments: int
) -> np.ndarray:
    """
    Apply Piecewise Aggregate Approximation.

    This implementation supports both divisible and non-divisible
    series lengths by using overlap-weighted segment means.

    Parameters
    ----------
    series:
        One-dimensional time series.

    n_segments:
        Number of PAA output segments.

    Returns
    -------
    numpy.ndarray
        One-dimensional PAA representation.
    """
    values = validate_univariate_series(series)

    if isinstance(n_segments, bool) or not isinstance(n_segments, Integral):
        raise TypeError("n_segments must be an integer.")

    n_segments = int(n_segments)

    if n_segments <= 0:
        raise ValueError("n_segments must be greater than zero.")

    if n_segments > values.size:
        raise ValueError(
            "n_segments cannot be greater than the number of observations."
        )

    if n_segments == values.size:
        return values.copy()

    segment_width = values.size / n_segments
    transformed = np.empty(n_segments, dtype=np.float64)

    for segment_index in range(n_segments):
        segment_start = segment_index * segment_width
        segment_end = (segment_index + 1) * segment_width

        first_value_index = int(np.floor(segment_start))
        last_value_index = int(np.ceil(segment_end))

        weighted_sum = 0.0

        for value_index in range(first_value_index, last_value_index):
            overlap_start = max(segment_start, float(value_index))
            overlap_end = min(segment_end, float(value_index + 1))
            overlap = overlap_end - overlap_start

            if overlap > 0:
                weighted_sum += values[value_index] * overlap

        transformed[segment_index] = weighted_sum / segment_width

    return transformed


@dataclass(frozen=True)
class PAATransformer:
    """
    Small reusable wrapper around PAA transformation.
    """

    n_segments: int

    def transform(
        self,
        series: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """
        Transform one univariate series into PAA segments.
        """
        return paa_transform(series=series, n_segments=self.n_segments)