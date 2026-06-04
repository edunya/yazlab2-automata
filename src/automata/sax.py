"""
Symbolic Aggregate approXimation (SAX).

SAX converts numerical PAA values into symbolic values by:
1. Applying train-fitted z-normalization.
2. Mapping normalized values through Gaussian breakpoints.

Leakage prevention rule:
The mean and standard deviation are learned only from training values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
from scipy.stats import norm

from src.automata.paa import validate_univariate_series


def generate_symbols(alphabet_size: int) -> np.ndarray:
    """
    Generate ordered SAX symbols: a, b, c, ...

    The project requires alphabet sizes 3, 4, 5 and 6,
    but this utility safely supports values from 2 to 26.
    """
    if not isinstance(alphabet_size, int) or isinstance(alphabet_size, bool):
        raise TypeError("alphabet_size must be an integer.")

    if alphabet_size < 2 or alphabet_size > 26:
        raise ValueError("alphabet_size must be between 2 and 26.")

    return np.asarray(
        [chr(ord("a") + index) for index in range(alphabet_size)],
        dtype=str
    )


def gaussian_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Compute standard Gaussian SAX breakpoints.

    For alphabet size 3, two breakpoints divide the normal
    distribution into three equally probable regions.
    """
    generate_symbols(alphabet_size)

    quantiles = np.arange(1, alphabet_size, dtype=np.float64) / alphabet_size

    return norm.ppf(quantiles)


@dataclass
class SAXDiscretizer:
    """
    Train-fitted SAX discretizer.

    Parameters
    ----------
    alphabet_size:
        Number of SAX symbols.

    Example
    -------
    alphabet_size=3 produces symbols: a, b, c
    """

    alphabet_size: int
    symbols_: np.ndarray = field(init=False)
    breakpoints_: np.ndarray = field(init=False)
    mean_: float | None = field(default=None, init=False)
    scale_: float | None = field(default=None, init=False)
    constant_training_series_: bool = field(default=False, init=False)
    is_fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.symbols_ = generate_symbols(self.alphabet_size)
        self.breakpoints_ = gaussian_breakpoints(self.alphabet_size)

    def fit(
        self,
        train_values: Sequence[float] | np.ndarray
    ) -> "SAXDiscretizer":
        """
        Fit z-normalization parameters from training values only.
        """
        values = validate_univariate_series(
            train_values,
            series_name="train_values"
        )

        self.mean_ = float(values.mean())

        calculated_scale = float(values.std(ddof=0))

        if np.isclose(calculated_scale, 0.0):
            self.scale_ = 1.0
            self.constant_training_series_ = True
        else:
            self.scale_ = calculated_scale
            self.constant_training_series_ = False

        self.is_fitted = True

        return self

    def transform(
        self,
        values: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """
        Convert numerical values into SAX symbols.
        """
        if not self.is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SAXDiscretizer must be fitted first.")

        numeric_values = validate_univariate_series(values)

        normalized_values = (numeric_values - self.mean_) / self.scale_

        symbol_indices = np.searchsorted(
            self.breakpoints_,
            normalized_values,
            side="right"
        )

        return self.symbols_[symbol_indices]

    def fit_transform(
        self,
        train_values: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """
        Fit using training values and transform the same values.
        """
        self.fit(train_values)
        return self.transform(train_values)

    def summary(self) -> Dict[str, Any]:
        """
        Return SAX settings for logging and reporting.
        """
        if not self.is_fitted:
            raise RuntimeError("SAXDiscretizer must be fitted first.")

        return {
            "alphabet_size": self.alphabet_size,
            "symbols": self.symbols_.tolist(),
            "breakpoints": self.breakpoints_.tolist(),
            "normalization": "train_zscore",
            "train_mean": self.mean_,
            "train_scale": self.scale_,
            "constant_training_series": self.constant_training_series_
        }