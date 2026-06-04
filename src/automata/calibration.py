"""
Threshold calibration utilities for probabilistic automata anomaly scores.

The threshold is selected using validation labels only.
The final test partition must never be used for threshold calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """
    Stores validation-based anomaly threshold calibration results.
    """

    threshold: float
    f1_score: float
    precision: float
    recall: float
    candidate_count: int
    validation_sample_count: int
    strategy: str = "validation_best_f1"
    tie_break: str = "higher_precision_then_higher_threshold"

    def summary(self) -> Dict[str, Any]:
        """
        Return calibration information for logging.
        """
        return {
            "threshold": self.threshold,
            "validation_f1_score": self.f1_score,
            "validation_precision": self.precision,
            "validation_recall": self.recall,
            "candidate_count": self.candidate_count,
            "validation_sample_count": self.validation_sample_count,
            "threshold_strategy": self.strategy,
            "threshold_tie_break": self.tie_break
        }


def validate_scores_and_labels(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate anomaly scores and binary ground-truth labels.
    """
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels)

    if score_values.ndim != 1 or label_values.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional.")

    if len(score_values) == 0:
        raise ValueError("scores must not be empty.")

    if len(score_values) != len(label_values):
        raise ValueError("scores and labels must have identical lengths.")

    if not np.isfinite(score_values).all():
        raise ValueError("scores contain non-finite values.")

    unique_labels = set(label_values.tolist())

    if not unique_labels.issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1.")

    return score_values, label_values.astype(int)


def predict_with_threshold(
    scores: Sequence[float] | np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Predict anomaly labels from automata anomaly scores.

    Higher anomaly score means less probable behavior.
    Therefore:

        score >= threshold -> anomaly label 1
    """
    score_values = np.asarray(scores, dtype=np.float64)

    if score_values.ndim != 1:
        raise ValueError("scores must be one-dimensional.")

    if not np.isfinite(score_values).all():
        raise ValueError("scores contain non-finite values.")

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    return (score_values >= threshold).astype(int)


def calibrate_f1_threshold(
    validation_scores: Sequence[float] | np.ndarray,
    validation_labels: Sequence[int] | np.ndarray
) -> ThresholdCalibrationResult:
    """
    Select threshold that maximizes validation F1-score.

    Tie-breaking
    ------------
    If multiple thresholds produce the same F1-score:
    1. Prefer the threshold with higher precision.
    2. If precision is also equal, prefer the higher threshold.

    This makes the final decision rule more conservative in equal-quality
    cases and avoids unnecessary false alarms.
    """
    scores, labels = validate_scores_and_labels(
        scores=validation_scores,
        labels=validation_labels
    )

    if set(labels.tolist()) != {0, 1}:
        raise ValueError(
            "Threshold calibration requires both normal and anomaly "
            "labels in the validation set."
        )

    candidate_thresholds = np.unique(scores)

    best_result: ThresholdCalibrationResult | None = None
    best_ranking: tuple[float, float, float] | None = None

    for threshold in candidate_thresholds:
        predictions = predict_with_threshold(
            scores=scores,
            threshold=float(threshold)
        )

        current_f1 = float(
            f1_score(labels, predictions, zero_division=0)
        )
        current_precision = float(
            precision_score(labels, predictions, zero_division=0)
        )
        current_recall = float(
            recall_score(labels, predictions, zero_division=0)
        )

        ranking = (
            current_f1,
            current_precision,
            float(threshold)
        )

        if best_ranking is None or ranking > best_ranking:
            best_ranking = ranking
            best_result = ThresholdCalibrationResult(
                threshold=float(threshold),
                f1_score=current_f1,
                precision=current_precision,
                recall=current_recall,
                candidate_count=int(len(candidate_thresholds)),
                validation_sample_count=int(len(labels))
            )

    if best_result is None:
        raise RuntimeError("Threshold calibration failed unexpectedly.")

    return best_result