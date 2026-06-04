"""
Tests for visualization utilities.

The Agg backend allows plot tests without opening GUI windows.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.automata.probabilistic_automata import ProbabilisticAutomata
from src.visualization.plot_automata_graph import plot_automata_graph
from src.visualization.plot_confusion_matrix import plot_confusion_matrix
from src.visualization.plot_metrics import (
    plot_metric_comparison,
    plot_runtime_comparison
)
from src.visualization.plot_roc_pr import (
    plot_precision_recall_curve,
    plot_roc_curve
)
from src.visualization.plot_transition_heatmap import (
    plot_transition_heatmap
)


def test_confusion_matrix_plot_is_saved(tmp_path):
    output_path = tmp_path / "confusion_matrix.png"

    figure, _ = plot_confusion_matrix(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        save_path=output_path
    )

    assert output_path.exists()

    plt.close(figure)


def test_roc_and_precision_recall_plots_are_saved(tmp_path):
    roc_path = tmp_path / "roc.png"
    pr_path = tmp_path / "pr.png"

    roc_figure, _ = plot_roc_curve(
        y_true=[0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.9],
        save_path=roc_path
    )

    pr_figure, _ = plot_precision_recall_curve(
        y_true=[0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.9],
        save_path=pr_path
    )

    assert roc_path.exists()
    assert pr_path.exists()

    plt.close(roc_figure)
    plt.close(pr_figure)


def test_metric_and_runtime_comparison_plots_are_saved(tmp_path):
    metric_path = tmp_path / "metric_comparison.png"
    runtime_path = tmp_path / "runtime_comparison.png"

    results_table = pd.DataFrame({
        "dataset": ["SKAB", "SKAB", "BATADAL", "BATADAL"],
        "model": ["lstm", "gru", "lstm", "gru"],
        "f1_score": [0.80, 0.82, 0.60, 0.63]
    })

    runtime_summary = pd.DataFrame({
        "model": ["lstm", "gru"],
        "training_seconds_mean": [10.0, 8.0]
    })

    metric_figure, _ = plot_metric_comparison(
        results_table=results_table,
        metric="f1_score",
        save_path=metric_path
    )

    runtime_figure, _ = plot_runtime_comparison(
        runtime_summary=runtime_summary,
        save_path=runtime_path
    )

    assert metric_path.exists()
    assert runtime_path.exists()

    plt.close(metric_figure)
    plt.close(runtime_figure)


def test_transition_heatmap_is_saved(tmp_path):
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit({
        "normal_run": ["aa", "ab", "aa", "ab"]
    })

    matrix = automata.transition_matrix()
    output_path = tmp_path / "transition_heatmap.png"

    figure, _ = plot_transition_heatmap(
        transition_matrix=matrix,
        save_path=output_path
    )

    assert output_path.exists()

    plt.close(figure)


def test_automata_graph_is_saved(tmp_path):
    automata = ProbabilisticAutomata(smoothing=1.0)

    automata.fit({
        "normal_run": ["aa", "ab", "aa", "ab"]
    })

    output_path = tmp_path / "automata_graph.png"

    figure, _ = plot_automata_graph(
        automata=automata,
        max_edges=10,
        save_path=output_path
    )

    assert output_path.exists()

    plt.close(figure)