"""
ROC and Precision-Recall curve visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)

from src.evaluation.metrics import (
    validate_binary_labels,
    validate_continuous_scores
)


def _validate_curve_inputs(
    y_true: Sequence[int],
    scores: Sequence[float]
):
    """
    Validate curve inputs and ensure both classes exist.
    """
    labels = validate_binary_labels(y_true, "y_true")
    validated_scores = validate_continuous_scores(scores)

    if len(labels) != len(validated_scores):
        raise ValueError("y_true and scores must have identical lengths.")

    if set(labels.tolist()) != {0, 1}:
        raise ValueError(
            "ROC and Precision-Recall curves require both binary classes."
        )

    return labels, validated_scores


def plot_roc_curve(
    y_true: Sequence[int],
    scores: Sequence[float],
    title: str = "ROC Curve",
    save_path: Optional[str | Path] = None,
    dpi: int = 300
):
    """
    Plot ROC curve from continuous anomaly scores.
    """
    labels, validated_scores = _validate_curve_inputs(y_true, scores)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        validated_scores
    )

    area_under_curve = roc_auc_score(labels, validated_scores)

    figure, axis = plt.subplots(figsize=(6, 5))

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {area_under_curve:.4f}"
    )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")

    axis.set_title(title)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return figure, axis


def plot_precision_recall_curve(
    y_true: Sequence[int],
    scores: Sequence[float],
    title: str = "Precision-Recall Curve",
    save_path: Optional[str | Path] = None,
    dpi: int = 300
):
    """
    Plot Precision-Recall curve from continuous anomaly scores.
    """
    labels, validated_scores = _validate_curve_inputs(y_true, scores)

    precision, recall, _ = precision_recall_curve(
        labels,
        validated_scores
    )

    average_precision = average_precision_score(
        labels,
        validated_scores
    )

    figure, axis = plt.subplots(figsize=(6, 5))

    axis.plot(
        recall,
        precision,
        label=f"Average Precision = {average_precision:.4f}"
    )

    axis.set_title(title)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="lower left")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return figure, axis