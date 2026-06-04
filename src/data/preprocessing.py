"""
Preprocessing utilities for deep learning and probabilistic automata models.

Leakage prevention rule:
All preprocessing objects must be fitted only on training data.
Validation and test partitions must only be transformed with fitted objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def validate_numeric_feature_frame(
    X: pd.DataFrame,
    expected_columns: List[str] | None = None
) -> None:
    """
    Validate a numerical feature dataframe.

    Parameters
    ----------
    X:
        Feature dataframe.

    expected_columns:
        Optional expected feature column ordering.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Features must be provided as a pandas DataFrame.")

    if X.empty:
        raise ValueError("Feature dataframe is empty.")

    if expected_columns is not None and list(X.columns) != expected_columns:
        raise ValueError(
            "Feature columns do not match the fitted preprocessing pipeline. "
            f"Expected {expected_columns}, found {list(X.columns)}."
        )

    non_numeric_columns = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "All feature columns must be numeric. "
            f"Non-numeric columns found: {non_numeric_columns}"
        )

    if X.isna().any().any():
        raise ValueError("Feature dataframe contains missing values.")


@dataclass
class DeepLearningPreprocessor:
    """
    StandardScaler based preprocessing for LSTM, GRU and 1D-CNN models.

    The scaler is fitted only on training features.
    """

    scaler: StandardScaler = field(default_factory=StandardScaler, init=False)
    feature_columns: List[str] = field(default_factory=list, init=False)
    is_fitted: bool = field(default=False, init=False)

    def fit(self, X_train: pd.DataFrame) -> "DeepLearningPreprocessor":
        """
        Fit StandardScaler using training data only.
        """
        validate_numeric_feature_frame(X_train)

        self.feature_columns = list(X_train.columns)
        self.scaler.fit(X_train[self.feature_columns])
        self.is_fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted training scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("DeepLearningPreprocessor must be fitted first.")

        validate_numeric_feature_frame(X, self.feature_columns)

        transformed_values = self.scaler.transform(X[self.feature_columns])

        return pd.DataFrame(
            transformed_values,
            columns=self.feature_columns,
            index=X.index
        )

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit using training data and transform the same training data.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def summary(self) -> Dict[str, Any]:
        """
        Return fitted scaler information for experiment logging.
        """
        if not self.is_fitted:
            raise RuntimeError("DeepLearningPreprocessor must be fitted first.")

        return {
            "preprocessor": "standard_scaler",
            "feature_columns": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "train_mean": self.scaler.mean_.tolist(),
            "train_scale": self.scaler.scale_.tolist()
        }


@dataclass
class AutomataPreprocessor:
    """
    Preprocessing for probabilistic automata.

    Steps
    -----
    1. StandardScaler fitted on training features only.
    2. PCA fitted on scaled training features only.
    3. One-dimensional PC1 series produced for all partitions.

    Note
    ----
    SAX-specific z-normalization of the PC1 time series will be implemented
    in the automata/SAX pipeline.
    """

    n_components: int = 1
    scaler: StandardScaler = field(default_factory=StandardScaler, init=False)
    pca: PCA = field(init=False)
    feature_columns: List[str] = field(default_factory=list, init=False)
    is_fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.n_components != 1:
            raise ValueError(
                "Automata preprocessing must produce exactly one PC1 component."
            )

        self.pca = PCA(n_components=self.n_components)

    def fit(self, X_train: pd.DataFrame) -> "AutomataPreprocessor":
        """
        Fit StandardScaler and PCA using training data only.
        """
        validate_numeric_feature_frame(X_train)

        self.feature_columns = list(X_train.columns)

        scaled_train = self.scaler.fit_transform(X_train[self.feature_columns])
        self.pca.fit(scaled_train)

        self.is_fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features into one-dimensional PC1 representation.
        """
        if not self.is_fitted:
            raise RuntimeError("AutomataPreprocessor must be fitted first.")

        validate_numeric_feature_frame(X, self.feature_columns)

        scaled_values = self.scaler.transform(X[self.feature_columns])
        pc1_values = self.pca.transform(scaled_values)

        return pd.DataFrame(
            pc1_values,
            columns=["PC1"],
            index=X.index
        )

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit using training data and transform the same training data.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def summary(self) -> Dict[str, Any]:
        """
        Return fitted preprocessing information for experiment logging.
        """
        if not self.is_fitted:
            raise RuntimeError("AutomataPreprocessor must be fitted first.")

        return {
            "preprocessor": "standard_scaler_plus_pca",
            "feature_columns": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "pca_n_components": self.n_components,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist()
        }


def build_automata_preprocessor_from_config(
    config: Dict[str, Any]
) -> AutomataPreprocessor:
    """
    Build AutomataPreprocessor from project configuration.
    """
    n_components = int(config["preprocessing"]["automata_n_components"])

    return AutomataPreprocessor(n_components=n_components)