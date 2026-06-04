"""
Metric and runtime comparison visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(
    results_table: pd.DataFrame,
    metric: str = "f1_score",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 300
):
    """
    Plot mean metric scores by model.

    If a dataset column exists, one group of bars is shown per dataset.
    """
    required_columns = {"model", metric}
    missing_columns = required_columns.difference(results_table.columns)

    if missing_columns:
        raise ValueError(
            f"Results table is missing columns: {sorted(missing_columns)}"
        )

    group_columns = ["model"]

    if "dataset" in results_table.columns:
        group_columns.insert(0, "dataset")

    summary = (
        results_table
        .groupby(group_columns, as_index=False)[metric]
        .mean()
    )

    if "dataset" in summary.columns:
        plot_table = summary.pivot(
            index="model",
            columns="dataset",
            values=metric
        )
    else:
        plot_table = summary.set_index("model")[[metric]]

    figure, axis = plt.subplots(figsize=(8, 5))

    plot_table.plot(
        kind="bar",
        ax=axis
    )

    axis.set_title(title or f"Model Comparison by {metric}")
    axis.set_xlabel("Model")
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=0)
    axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return figure, axis


def plot_runtime_comparison(
    runtime_summary: pd.DataFrame,
    time_column: str = "training_seconds_mean",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 300
):
    """
    Plot mean runtime values by model.
    """
    required_columns = {"model", time_column}
    missing_columns = required_columns.difference(runtime_summary.columns)

    if missing_columns:
        raise ValueError(
            f"Runtime table is missing columns: {sorted(missing_columns)}"
        )

    grouped = (
        runtime_summary
        .groupby("model", as_index=False)[time_column]
        .mean()
        .set_index("model")
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    grouped.plot(
        kind="bar",
        legend=False,
        ax=axis
    )

    axis.set_title(title or f"Runtime Comparison by {time_column}")
    axis.set_xlabel("Model")
    axis.set_ylabel("Seconds")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return figure, axis