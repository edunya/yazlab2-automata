"""
Scenario utilities for robustness and automata parameter analysis.

This module intentionally does not train models. It prepares controlled
scenario inputs and reporting summaries that will later be used by the
batch experiment orchestrator.

Robustness principles
---------------------
- Gaussian noise is applied only to test observations.
- Noise scale is derived only from training observations.
- Clean trained models must later be reused for noisy test evaluation.
- Unseen symbolic-state analysis applies to probabilistic automata only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.automata.explainability import AutomataDecisionExplanation
from src.data.splitting import DatasetSplit
from src.evaluation.metrics import evaluate_binary_scores


@dataclass(frozen=True)
class GaussianNoiseSetting:
    """
    Defines one Gaussian noise robustness scenario.

    Noise level is expressed as a fraction of the training feature
    standard deviation.

    Examples
    --------
    level=0.01 means noise standard deviation equals 1% of
    each feature's training standard deviation.
    """

    level: float
    seed: int
    apply_to: str = "test_only"
    scale_reference: str = "training_feature_standard_deviation"
    retrain_model: bool = False

    @property
    def scenario_name(self) -> str:
        """
        Return stable scenario name for logs and result tables.
        """
        return f"gaussian_noise_{self.level:.2f}"

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-compatible scenario representation.
        """
        return {
            "scenario": self.scenario_name,
            "noise_level": float(self.level),
            "noise_seed": int(self.seed),
            "apply_to": self.apply_to,
            "scale_reference": self.scale_reference,
            "retrain_model": self.retrain_model
        }


@dataclass(frozen=True)
class AutomataParameterSetting:
    """
    One automata parameter-analysis configuration.
    """

    window_size: int
    alphabet_size: int
    context_length: int

    @property
    def scenario_name(self) -> str:
        """
        Return stable scenario name for logs.
        """
        return (
            f"automata_parameter_w{self.window_size}"
            f"_a{self.alphabet_size}"
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-compatible setting representation.
        """
        return {
            "scenario": self.scenario_name,
            "context_length": int(self.context_length),
            "window_size": int(self.window_size),
            "alphabet_size": int(self.alphabet_size)
        }


@dataclass(frozen=True)
class UnseenPatternAnalysisSummary:
    """
    Summary of automata decisions involving unseen symbolic states.
    """

    total_decisions: int
    seen_only_decisions: int
    unseen_involved_decisions: int
    total_state_occurrences: int
    unseen_state_occurrences: int
    unseen_decision_ratio: float
    unseen_state_occurrence_ratio: float
    seen_only_metrics: Optional[Dict[str, Any]]
    unseen_involved_metrics: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-compatible unseen analysis summary.
        """
        return {
            "total_decisions": self.total_decisions,
            "seen_only_decisions": self.seen_only_decisions,
            "unseen_involved_decisions": self.unseen_involved_decisions,
            "total_state_occurrences": self.total_state_occurrences,
            "unseen_state_occurrences": self.unseen_state_occurrences,
            "unseen_decision_ratio": self.unseen_decision_ratio,
            "unseen_state_occurrence_ratio": self.unseen_state_occurrence_ratio,
            "seen_only_metrics": self.seen_only_metrics,
            "unseen_involved_metrics": self.unseen_involved_metrics
        }


def _validate_numeric_feature_frame(X: pd.DataFrame) -> None:
    """
    Validate feature dataframe for scenario generation.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X must not be empty.")

    non_numeric_columns = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "Gaussian noise can be applied only to numeric features. "
            f"Non-numeric columns found: {non_numeric_columns}"
        )

    if X.isna().any().any():
        raise ValueError("X contains missing values.")


def build_gaussian_noise_settings(
    config: Dict[str, Any]
) -> list[GaussianNoiseSetting]:
    """
    Create configured Gaussian noise scenarios.
    """
    noise_config = config["experiments"]["robustness"]["gaussian_noise"]

    levels = noise_config["levels"]
    seed = int(noise_config["noise_seed"])

    settings = []

    for level in levels:
        if not isinstance(level, Real):
            raise TypeError("Gaussian noise levels must be numerical.")

        level = float(level)

        if level < 0:
            raise ValueError("Gaussian noise levels must be non-negative.")

        settings.append(
            GaussianNoiseSetting(
                level=level,
                seed=seed,
                apply_to=noise_config["apply_to"],
                scale_reference=noise_config["scale_reference"],
                retrain_model=bool(noise_config["retrain_model"])
            )
        )

    return settings


def inject_gaussian_noise_into_test_rows(
    X: pd.DataFrame,
    split: DatasetSplit,
    setting: GaussianNoiseSetting
) -> pd.DataFrame:
    """
    Add Gaussian noise only to test rows.

    The noise scale for each feature is computed from training rows only:

        noise_std(feature) =
            setting.level * std(training_feature)

    Train and validation rows remain unchanged.
    """
    _validate_numeric_feature_frame(X)

    if setting.apply_to != "test_only":
        raise ValueError(
            "This project permits Gaussian noise only on the test partition."
        )

    if setting.scale_reference != "training_feature_standard_deviation":
        raise ValueError(
            "This project requires train-derived feature standard deviation "
            "for Gaussian noise scaling."
        )

    if setting.retrain_model:
        raise ValueError(
            "Robustness evaluation must reuse the clean trained model. "
            "retrain_model must be False."
        )

    if setting.level < 0:
        raise ValueError("Noise level must be non-negative.")

    noisy_X = X.copy()

    if setting.level == 0:
        return noisy_X

    train_values = X.iloc[split.train_indices].to_numpy(dtype=np.float64)
    test_values = X.iloc[split.test_indices].to_numpy(dtype=np.float64)

    train_standard_deviation = train_values.std(axis=0, ddof=0)

    generator = np.random.default_rng(setting.seed)

    random_noise = generator.normal(
        loc=0.0,
        scale=1.0,
        size=test_values.shape
    )

    scaled_noise = (
        random_noise
        * train_standard_deviation.reshape(1, -1)
        * setting.level
    )

    noisy_test_values = test_values + scaled_noise

    noisy_X.iloc[split.test_indices] = noisy_test_values

    return noisy_X


def create_automata_parameter_grid(
    config: Dict[str, Any]
) -> list[AutomataParameterSetting]:
    """
    Create all configured automata parameter combinations.
    """
    parameter_config = (
        config["experiments"]["parameter_analysis"]["automata"]
    )

    context_length = int(parameter_config["context_length_fixed"])
    window_sizes = parameter_config["window_size_values"]
    alphabet_sizes = parameter_config["alphabet_size_values"]

    settings = []

    for window_size in window_sizes:
        for alphabet_size in alphabet_sizes:
            window_size = int(window_size)
            alphabet_size = int(alphabet_size)

            if window_size <= 0:
                raise ValueError("window_size must be greater than zero.")

            if window_size > context_length:
                raise ValueError(
                    "window_size cannot exceed fixed context_length."
                )

            if alphabet_size < 2:
                raise ValueError("alphabet_size must be at least 2.")

            settings.append(
                AutomataParameterSetting(
                    window_size=window_size,
                    alphabet_size=alphabet_size,
                    context_length=context_length
                )
            )

    return settings


def build_config_for_automata_setting(
    config: Dict[str, Any],
    setting: AutomataParameterSetting
) -> Dict[str, Any]:
    """
    Return a copied project config configured for one automata setting.

    The input configuration is never modified in place.
    """
    adjusted_config = deepcopy(config)

    adjusted_config["automata"]["context_length"] = int(
        setting.context_length
    )
    adjusted_config["automata"]["default_window_size"] = int(
        setting.window_size
    )
    adjusted_config["automata"]["default_alphabet_size"] = int(
        setting.alphabet_size
    )

    return adjusted_config


def summarize_unseen_automata_decisions(
    explanations: Sequence[AutomataDecisionExplanation],
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    zero_division: int = 0
) -> UnseenPatternAnalysisSummary:
    """
    Summarize automata performance for seen-only and unseen-involved decisions.

    A decision is defined as unseen-involved if at least one state in its
    scored path was unseen during automata training and required mapping.
    """
    explanation_list = list(explanations)
    label_values = np.asarray(labels, dtype=int)
    score_values = np.asarray(scores, dtype=np.float64)

    if not explanation_list:
        raise ValueError("explanations must not be empty.")

    if len(explanation_list) != len(label_values):
        raise ValueError(
            "explanations and labels must have identical lengths."
        )

    if len(explanation_list) != len(score_values):
        raise ValueError(
            "explanations and scores must have identical lengths."
        )

    unseen_mask = np.asarray(
        [
            explanation.unseen_state_count > 0
            for explanation in explanation_list
        ],
        dtype=bool
    )

    seen_mask = ~unseen_mask

    total_state_occurrences = int(
        sum(len(explanation.state_mappings) for explanation in explanation_list)
    )

    unseen_state_occurrences = int(
        sum(explanation.unseen_state_count for explanation in explanation_list)
    )

    def evaluate_subset(mask: np.ndarray) -> Optional[Dict[str, Any]]:
        if not mask.any():
            return None

        return evaluate_binary_scores(
            y_true=label_values[mask],
            scores=score_values[mask],
            threshold=threshold,
            score_name="automata_anomaly_score",
            zero_division=zero_division
        ).as_dict()

    return UnseenPatternAnalysisSummary(
        total_decisions=int(len(explanation_list)),
        seen_only_decisions=int(seen_mask.sum()),
        unseen_involved_decisions=int(unseen_mask.sum()),
        total_state_occurrences=total_state_occurrences,
        unseen_state_occurrences=unseen_state_occurrences,
        unseen_decision_ratio=float(unseen_mask.mean()),
        unseen_state_occurrence_ratio=float(
            unseen_state_occurrences / total_state_occurrences
        ),
        seen_only_metrics=evaluate_subset(seen_mask),
        unseen_involved_metrics=evaluate_subset(unseen_mask)
    )