"""
Confusion matrix visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import validate_binary_labels


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
    dpi: int = 300
):
    """
    Plot binary confusion matrix.

    Returns
    -------
    tuple
        matplotlib figure and axes objects.
    """
    true_labels = validate_binary_labels(y_true, "y_true")
    predicted_labels = validate_binary_labels(y_pred, "y_pred")

    if len(true_labels) != len(predicted_labels):
        raise ValueError("y_true and y_pred must have identical lengths.")

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    figure, axis = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=axis
    )

    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return figure, axis