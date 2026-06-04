"""
Transition probability heatmap visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_transition_heatmap(
    transition_matrix: pd.DataFrame,
    title: str = "Automata Transition Probability Heatmap",
    save_path: Optional[str | Path] = None,
    dpi: int = 300
):
    """
    Plot transition probability matrix as a heatmap.
    """
    if not isinstance(transition_matrix, pd.DataFrame):
        raise TypeError("transition_matrix must be a pandas DataFrame.")

    if transition_matrix.empty:
        raise ValueError("transition_matrix must not be empty.")

    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("transition_matrix must be square.")

    figure_size = max(6, min(14, transition_matrix.shape[0] * 0.35))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))

    sns.heatmap(
        transition_matrix,
        ax=axis,
        square=True,
        cbar=True
    )

    axis.set_title(title)
    axis.set_xlabel("Next State")
    axis.set_ylabel("Current State")

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return figure, axis