"""
Evaluation metrics for binary anomaly detection models.

This module supports both:
- Deep learning probability scores.
- Probabilistic automata anomaly scores.

Common convention:
Higher score means a higher likelihood of anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def validate_binary_labels(
    labels: Sequence[int] | np.ndarray,
    label_name: str = "labels"
) -> np.ndarray:
    """
    Validate one-dimensional binary labels.
    """
    values = np.asarray(labels)

    if values.ndim != 1:
        raise ValueError(f"{label_name} must be one-dimensional.")

    if len(values) == 0:
        raise ValueError(f"{label_name} must not be empty.")

    unique_values = set(values.tolist())

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{label_name} must contain only binary values 0 and 1."
        )

    return values.astype(int)


def validate_continuous_scores(
    scores: Sequence[float] | np.ndarray
) -> np.ndarray:
    """
    Validate continuous anomaly scores.
    """
    values = np.asarray(scores, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional.")

    if len(values) == 0:
        raise ValueError("scores must not be empty.")

    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values.")

    return values


def logits_to_probabilities(
    logits: Sequence[float] | np.ndarray
) -> np.ndarray:
    """
    Convert deep learning raw logits into anomaly probabilities.
    """
    validated_logits = validate_continuous_scores(logits)

    return expit(validated_logits)


def predictions_from_scores(
    scores: Sequence[float] | np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Convert anomaly scores into binary predictions.

    Convention
    ----------
    score >= threshold -> anomaly class 1
    """
    validated_scores = validate_continuous_scores(scores)

    if not isinstance(threshold, Real):
        raise TypeError("threshold must be numerical.")

    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    return (validated_scores >= threshold).astype(int)


@dataclass(frozen=True)
class BinaryClassificationResult:
    """
    Stores evaluation results for one binary anomaly detector.
    """

    threshold: float
    score_name: str
    sample_count: int
    anomaly_count: int
    predicted_anomaly_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    average_precision: Optional[float]
    confusion_matrix: list[list[int]]

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-compatible evaluation results.
        """
        true_negative, false_positive = self.confusion_matrix[0]
        false_negative, true_positive = self.confusion_matrix[1]

        return {
            "threshold": self.threshold,
            "score_name": self.score_name,
            "sample_count": self.sample_count,
            "anomaly_count": self.anomaly_count,
            "predicted_anomaly_count": self.predicted_anomaly_count,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
            "confusion_matrix": self.confusion_matrix
        }


def evaluate_binary_scores(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    score_name: str = "anomaly_score",
    zero_division: int = 0
) -> BinaryClassificationResult:
    """
    Evaluate binary anomaly predictions from continuous scores.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.

    scores:
        Continuous anomaly scores. Higher values must indicate anomaly.

    threshold:
        Decision threshold.

    score_name:
        Name recorded in output, such as:
        - "probability"
        - "automata_anomaly_score"

    zero_division:
        Value used by precision/recall/F1 when no positive prediction exists.
    """
    validated_labels = validate_binary_labels(y_true, label_name="y_true")
    validated_scores = validate_continuous_scores(scores)

    if len(validated_labels) != len(validated_scores):
        raise ValueError("y_true and scores must have identical lengths.")

    predictions = predictions_from_scores(
        scores=validated_scores,
        threshold=threshold
    )

    matrix = confusion_matrix(
        validated_labels,
        predictions,
        labels=[0, 1]
    )

    unique_classes = set(validated_labels.tolist())

    if unique_classes == {0, 1}:
        roc_auc = float(
            roc_auc_score(validated_labels, validated_scores)
        )
        average_precision = float(
            average_precision_score(validated_labels, validated_scores)
        )
    else:
        roc_auc = None
        average_precision = None

    return BinaryClassificationResult(
        threshold=float(threshold),
        score_name=score_name,
        sample_count=int(len(validated_labels)),
        anomaly_count=int(validated_labels.sum()),
        predicted_anomaly_count=int(predictions.sum()),
        accuracy=float(accuracy_score(validated_labels, predictions)),
        precision=float(
            precision_score(
                validated_labels,
                predictions,
                zero_division=zero_division
            )
        ),
        recall=float(
            recall_score(
                validated_labels,
                predictions,
                zero_division=zero_division
            )
        ),
        f1_score=float(
            f1_score(
                validated_labels,
                predictions,
                zero_division=zero_division
            )
        ),
        roc_auc=roc_auc,
        average_precision=average_precision,
        confusion_matrix=matrix.astype(int).tolist()
    )