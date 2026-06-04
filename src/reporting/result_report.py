"""
Result-table and figure-package generation.

This module consumes already exported experiment results and artifacts.
It never trains models and never executes dataset pipelines.

Expected input structure
------------------------
results_dir/
    results_raw.csv
    artifacts/
        <task_id>/
            original/
                predictions.csv
            gaussian_noise_0.05/
                predictions.csv
            transition_matrix.csv
            observed_transition_edges.csv

Generated output structure
--------------------------
report_package/
    report_summary.json
    tables/
    figures/
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

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
from src.reporting.statistical_analysis import (
    build_paired_wilcoxon_summary
)


BASELINE_TASK_TYPES = {
    "deep_learning_robustness",
    "automata_robustness"
}


def _safe_name(value: str) -> str:
    """
    Convert a label into a filesystem-safe filename component.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_json(data: Any, file_path: Path) -> None:
    """
    Write JSON using UTF-8 and common NumPy-safe conversions.
    """
    def json_default(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            data,
            json_file,
            ensure_ascii=False,
            indent=2,
            default=json_default
        )


def load_results_table(results_dir: str | Path) -> pd.DataFrame:
    """
    Load raw flat result rows exported by controlled execution.
    """
    results_path = Path(results_dir) / "results_raw.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Raw results file was not found: {results_path}"
        )

    results_table = pd.read_csv(results_path)

    required_columns = {
        "task_id",
        "task_type",
        "dataset",
        "model",
        "scenario",
        "f1_score"
    }

    missing_columns = required_columns.difference(results_table.columns)

    if missing_columns:
        raise ValueError(
            f"Results table is missing columns: {sorted(missing_columns)}"
        )

    return results_table


def _baseline_rows(results_table: pd.DataFrame) -> pd.DataFrame:
    """
    Return original/noise rows belonging to baseline model tasks.
    """
    return results_table[
        results_table["task_type"].isin(BASELINE_TASK_TYPES)
    ].copy()


def _fill_standard_deviation_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN standard deviation values created by one-run groups.
    """
    standard_deviation_columns = [
        column
        for column in table.columns
        if column.endswith("_std")
    ]

    if standard_deviation_columns:
        table[standard_deviation_columns] = (
            table[standard_deviation_columns].fillna(0.0)
        )

    return table


def build_original_model_summary(
    results_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize clean/original baseline performance by dataset and model.
    """
    original_rows = _baseline_rows(results_table)

    original_rows = original_rows[
        original_rows["scenario"] == "original"
    ]

    if original_rows.empty:
        raise ValueError("No original baseline result rows were found.")

    available_metrics = [
        metric
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
            "training_seconds",
            "inference_seconds"
        ]
        if metric in original_rows.columns
    ]

    aggregations: Dict[str, tuple[str, str]] = {
        "run_count": ("task_id", "nunique")
    }

    for metric in available_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    summary = (
        original_rows
        .groupby(["dataset", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["dataset", "f1_score_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return _fill_standard_deviation_columns(summary)


def build_robustness_degradation_summary(
    results_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize F1 degradation under Gaussian-noise scenarios.

    Each noisy result is compared only with the original result generated
    by the same task. This preserves fold/seed pairing.
    """
    baseline_rows = _baseline_rows(results_table)

    original_scores = (
        baseline_rows[baseline_rows["scenario"] == "original"][
            ["task_id", "f1_score"]
        ]
        .rename(columns={"f1_score": "original_f1_score"})
    )

    noise_rows = baseline_rows[
        baseline_rows["scenario"].str.startswith("gaussian_noise_")
    ].copy()

    if noise_rows.empty:
        return pd.DataFrame()

    noise_rows = noise_rows.merge(
        original_scores,
        on="task_id",
        how="left",
        validate="many_to_one"
    )

    if noise_rows["original_f1_score"].isna().any():
        raise ValueError(
            "A Gaussian-noise result row has no matching original row."
        )

    noise_rows["f1_score_drop_from_original"] = (
        noise_rows["original_f1_score"] - noise_rows["f1_score"]
    )

    summary = (
        noise_rows
        .groupby(["dataset", "model", "scenario"], as_index=False)
        .agg(
            run_count=("task_id", "nunique"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            f1_score_drop_mean=("f1_score_drop_from_original", "mean"),
            f1_score_drop_std=("f1_score_drop_from_original", "std")
        )
        .sort_values(["dataset", "model", "scenario"])
        .reset_index(drop=True)
    )

    return _fill_standard_deviation_columns(summary)


def build_automata_parameter_summary(
    results_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize automata parameter-analysis rows.
    """
    parameter_rows = results_table[
        results_table["task_type"] == "automata_parameter_analysis"
    ].copy()

    if parameter_rows.empty:
        return pd.DataFrame()

    required_columns = {"window_size", "alphabet_size"}

    missing_columns = required_columns.difference(parameter_rows.columns)

    if missing_columns:
        raise ValueError(
            "Automata parameter rows are missing columns: "
            f"{sorted(missing_columns)}"
        )

    available_metrics = [
        metric
        for metric in [
            "f1_score",
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
            "training_seconds",
            "inference_seconds",
            "state_count",
            "observed_transition_count",
            "possible_transition_count",
            "transition_density"
        ]
        if metric in parameter_rows.columns
    ]

    aggregations: Dict[str, tuple[str, str]] = {
        "run_count": ("task_id", "nunique")
    }

    for metric in available_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    summary = (
        parameter_rows
        .groupby(
            ["dataset", "window_size", "alphabet_size"],
            as_index=False
        )
        .agg(**aggregations)
        .sort_values(
            ["dataset", "f1_score_mean"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )

    return _fill_standard_deviation_columns(summary)


def build_unseen_analysis_summary(
    results_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize automata unseen-state behavior by dataset and scenario.
    """
    automata_rows = results_table[
        results_table["task_type"] == "automata_robustness"
    ].copy()

    required_columns = {
        "unseen_involved_decisions",
        "unseen_decision_ratio",
        "unseen_state_occurrences",
        "unseen_state_occurrence_ratio"
    }

    if automata_rows.empty or not required_columns.issubset(
        automata_rows.columns
    ):
        return pd.DataFrame()

    summary = (
        automata_rows
        .groupby(["dataset", "scenario"], as_index=False)
        .agg(
            run_count=("task_id", "nunique"),
            unseen_involved_decisions_mean=(
                "unseen_involved_decisions",
                "mean"
            ),
            unseen_decision_ratio_mean=("unseen_decision_ratio", "mean"),
            unseen_decision_ratio_std=("unseen_decision_ratio", "std"),
            unseen_state_occurrences_mean=(
                "unseen_state_occurrences",
                "mean"
            ),
            unseen_state_occurrence_ratio_mean=(
                "unseen_state_occurrence_ratio",
                "mean"
            ),
            unseen_state_occurrence_ratio_std=(
                "unseen_state_occurrence_ratio",
                "std"
            )
        )
        .sort_values(["dataset", "scenario"])
        .reset_index(drop=True)
    )

    return _fill_standard_deviation_columns(summary)


def load_pooled_prediction_table(
    results_table: pd.DataFrame,
    artifacts_dir: str | Path,
    dataset: str,
    model: str,
    scenario: str = "original"
) -> pd.DataFrame:
    """
    Load available sample-level predictions for one dataset/model/scenario.

    Deep learning probabilities are pooled directly.

    Automata paths may use different calibrated thresholds across folds.
    Therefore an additional curve_score column is defined as:

        curve_score = score - threshold

    for automata, which preserves the anomaly direction relative to each
    calibrated decision threshold.
    """
    artifacts_path = Path(artifacts_dir)

    matching_rows = _baseline_rows(results_table)

    matching_rows = matching_rows[
        (matching_rows["dataset"] == dataset)
        & (matching_rows["model"] == model)
        & (matching_rows["scenario"] == scenario)
    ]

    prediction_parts = []

    for task_id in matching_rows["task_id"].drop_duplicates().tolist():
        prediction_path = (
            artifacts_path
            / task_id
            / scenario
            / "predictions.csv"
        )

        if not prediction_path.exists():
            continue

        prediction_table = pd.read_csv(prediction_path)
        prediction_table["task_id"] = task_id
        prediction_parts.append(prediction_table)

    if not prediction_parts:
        return pd.DataFrame()

    pooled_predictions = pd.concat(
        prediction_parts,
        ignore_index=True
    )

    required_columns = {
        "y_true",
        "score",
        "threshold",
        "y_pred"
    }

    missing_columns = required_columns.difference(pooled_predictions.columns)

    if missing_columns:
        raise ValueError(
            "Prediction artifacts are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if model == "automata":
        pooled_predictions["curve_score"] = (
            pooled_predictions["score"]
            - pooled_predictions["threshold"]
        )
    else:
        pooled_predictions["curve_score"] = pooled_predictions["score"]

    return pooled_predictions


def plot_parameter_heatmap(
    parameter_summary: pd.DataFrame,
    dataset: str,
    save_path: str | Path,
    dpi: int = 300
):
    """
    Plot mean F1-score across automata window/alphabet combinations.
    """
    dataset_rows = parameter_summary[
        parameter_summary["dataset"] == dataset
    ]

    if dataset_rows.empty:
        raise ValueError(
            f"No automata parameter results found for dataset: {dataset}"
        )

    heatmap_table = dataset_rows.pivot(
        index="window_size",
        columns="alphabet_size",
        values="f1_score_mean"
    )

    figure, axis = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        heatmap_table,
        annot=True,
        fmt=".4f",
        ax=axis
    )

    axis.set_title(f"{dataset} Automata Parameter Analysis — Mean F1")
    axis.set_xlabel("Alphabet Size")
    axis.set_ylabel("PAA / SAX Word Size")

    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return figure, axis

def plot_transition_density_heatmap(
    parameter_summary: pd.DataFrame,
    dataset: str,
    save_path: str | Path,
    dpi: int = 300
):
    """
    Plot mean transition density across automata parameter combinations.
    """
    dataset_rows = parameter_summary[
        parameter_summary["dataset"] == dataset
    ]

    if dataset_rows.empty:
        raise ValueError(
            f"No automata parameter results found for dataset: {dataset}"
        )

    if "transition_density_mean" not in dataset_rows.columns:
        raise ValueError(
            "Parameter summary does not contain transition_density_mean."
        )

    heatmap_table = dataset_rows.pivot(
        index="window_size",
        columns="alphabet_size",
        values="transition_density_mean"
    )

    figure, axis = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        heatmap_table,
        annot=True,
        fmt=".4f",
        ax=axis
    )

    axis.set_title(
        f"{dataset} Automata Parameter Analysis — Transition Density"
    )
    axis.set_xlabel("Alphabet Size")
    axis.set_ylabel("PAA / SAX Word Size")

    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return figure, axis

def plot_automata_graph_from_edges(
    edge_table: pd.DataFrame,
    title: str,
    save_path: str | Path,
    max_edges: int = 30,
    dpi: int = 300
):
    """
    Plot an automata graph from exported observed transition edges.
    """
    required_columns = {
        "from_state",
        "to_state",
        "observed_count",
        "probability"
    }

    missing_columns = required_columns.difference(edge_table.columns)

    if missing_columns:
        raise ValueError(
            f"Observed-edge table is missing columns: {sorted(missing_columns)}"
        )

    selected_edges = (
        edge_table
        .sort_values(
            ["probability", "observed_count"],
            ascending=[False, False]
        )
        .head(max_edges)
    )

    graph = nx.DiGraph()

    for _, row in selected_edges.iterrows():
        graph.add_edge(
            row["from_state"],
            row["to_state"],
            probability=float(row["probability"])
        )

    figure, axis = plt.subplots(figsize=(10, 8))

    positions = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(graph, positions, node_size=850, ax=axis)
    nx.draw_networkx_labels(graph, positions, font_size=8, ax=axis)
    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        ax=axis
    )

    edge_labels = {
        (source, target): f"{attributes['probability']:.3f}"
        for source, target, attributes in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=7,
        ax=axis
    )

    axis.set_title(title)
    axis.axis("off")

    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return figure, axis


def generate_result_report_package(
    results_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    dpi: int = 300,
    automata_graph_max_edges: int = 30,
    statistical_metric: str = "f1_score",
    statistical_models: Optional[list[str]] = None,
    statistical_scenario: str = "original",
    statistical_alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Generate report-ready tables and figures from completed experiment output.

    This function never runs model training.
    """
    results_path = Path(results_dir)
    report_path = (
        Path(output_dir)
        if output_dir is not None
        else results_path / "report_package"
    )

    tables_dir = report_path / "tables"
    figures_dir = report_path / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = results_path / "artifacts"
    results_table = load_results_table(results_path)

    table_paths: Dict[str, Path] = {}
    figure_paths: Dict[str, Path] = {}

    original_summary = build_original_model_summary(results_table)
    original_summary_path = tables_dir / "original_model_summary.csv"
    original_summary.to_csv(original_summary_path, index=False)
    table_paths["original_model_summary"] = original_summary_path

    statistical_required_columns = {
        "task_type",
        "dataset",
        "model",
        "scenario",
        "fold",
        "seed",
        statistical_metric
    }

    if statistical_required_columns.issubset(results_table.columns):
        statistical_summary = build_paired_wilcoxon_summary(
            results_table=results_table,
            metric=statistical_metric,
            compared_models=(
                statistical_models
                if statistical_models is not None
                else ["lstm", "gru", "cnn1d"]
            ),
            scenario=statistical_scenario,
            alpha=statistical_alpha
        )

        if not statistical_summary.empty:
            statistical_path = (
                tables_dir / "statistical_significance_summary.csv"
            )
            statistical_summary.to_csv(statistical_path, index=False)
            table_paths["statistical_significance_summary"] = statistical_path

    robustness_summary = build_robustness_degradation_summary(results_table)

    if not robustness_summary.empty:
        robustness_path = tables_dir / "robustness_degradation_summary.csv"
        robustness_summary.to_csv(robustness_path, index=False)
        table_paths["robustness_degradation_summary"] = robustness_path

    parameter_summary = build_automata_parameter_summary(results_table)

    if not parameter_summary.empty:
        parameter_path = tables_dir / "automata_parameter_summary.csv"
        parameter_summary.to_csv(parameter_path, index=False)
        table_paths["automata_parameter_summary"] = parameter_path

    unseen_summary = build_unseen_analysis_summary(results_table)

    if not unseen_summary.empty:
        unseen_path = tables_dir / "automata_unseen_summary.csv"
        unseen_summary.to_csv(unseen_path, index=False)
        table_paths["automata_unseen_summary"] = unseen_path

    original_baseline_rows = _baseline_rows(results_table)

    original_baseline_rows = original_baseline_rows[
        original_baseline_rows["scenario"] == "original"
    ]

    metric_figure_path = figures_dir / "original_f1_model_comparison.png"

    metric_figure, _ = plot_metric_comparison(
        results_table=original_baseline_rows,
        metric="f1_score",
        title="Original Scenario — Mean F1-Score Comparison",
        save_path=metric_figure_path,
        dpi=dpi
    )

    plt.close(metric_figure)
    figure_paths["original_f1_model_comparison"] = metric_figure_path

    if "training_seconds" in original_baseline_rows.columns:
        runtime_figure_path = figures_dir / "original_training_runtime.png"

        runtime_figure, _ = plot_runtime_comparison(
            runtime_summary=original_baseline_rows,
            time_column="training_seconds",
            title="Original Scenario — Training Runtime Comparison",
            save_path=runtime_figure_path,
            dpi=dpi
        )

        plt.close(runtime_figure)
        figure_paths["original_training_runtime"] = runtime_figure_path

    baseline_pairs = (
        original_baseline_rows[["dataset", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for dataset, model in baseline_pairs:
        pooled_predictions = load_pooled_prediction_table(
            results_table=results_table,
            artifacts_dir=artifacts_dir,
            dataset=dataset,
            model=model,
            scenario="original"
        )

        if pooled_predictions.empty:
            continue

        figure_prefix = _safe_name(f"{dataset}__{model}__original")

        confusion_path = (
            figures_dir / f"{figure_prefix}__confusion_matrix.png"
        )

        confusion_figure, _ = plot_confusion_matrix(
            y_true=pooled_predictions["y_true"].to_numpy(),
            y_pred=pooled_predictions["y_pred"].to_numpy(),
            title=f"{dataset} — {model} — Pooled Original Confusion Matrix",
            save_path=confusion_path,
            dpi=dpi
        )

        plt.close(confusion_figure)
        figure_paths[f"{figure_prefix}_confusion_matrix"] = confusion_path

        if set(pooled_predictions["y_true"].tolist()) == {0, 1}:
            roc_path = figures_dir / f"{figure_prefix}__roc_curve.png"
            pr_path = (
                figures_dir
                / f"{figure_prefix}__precision_recall_curve.png"
            )

            curve_scores = pooled_predictions["curve_score"].to_numpy()

            roc_figure, _ = plot_roc_curve(
                y_true=pooled_predictions["y_true"].to_numpy(),
                scores=curve_scores,
                title=f"{dataset} — {model} — Pooled Original ROC Curve",
                save_path=roc_path,
                dpi=dpi
            )

            pr_figure, _ = plot_precision_recall_curve(
                y_true=pooled_predictions["y_true"].to_numpy(),
                scores=curve_scores,
                title=(
                    f"{dataset} — {model} — "
                    "Pooled Original Precision-Recall Curve"
                ),
                save_path=pr_path,
                dpi=dpi
            )

            plt.close(roc_figure)
            plt.close(pr_figure)

            figure_paths[f"{figure_prefix}_roc_curve"] = roc_path
            figure_paths[f"{figure_prefix}_precision_recall_curve"] = pr_path

    if not parameter_summary.empty:
        for dataset in parameter_summary["dataset"].unique().tolist():
            parameter_heatmap_path = (
                figures_dir
                / f"{_safe_name(dataset)}__automata_parameter_heatmap.png"
            )

            parameter_figure, _ = plot_parameter_heatmap(
                parameter_summary=parameter_summary,
                dataset=dataset,
                save_path=parameter_heatmap_path,
                dpi=dpi
            )

            plt.close(parameter_figure)
            figure_paths[
                f"{_safe_name(dataset)}_automata_parameter_heatmap"
            ] = parameter_heatmap_path

            if "transition_density_mean" in parameter_summary.columns:
                density_heatmap_path = (
                    figures_dir
                    / (
                        f"{_safe_name(dataset)}"
                        "__automata_transition_density_heatmap.png"
                    )
                )

                density_figure, _ = plot_transition_density_heatmap(
                    parameter_summary=parameter_summary,
                    dataset=dataset,
                    save_path=density_heatmap_path,
                    dpi=dpi
                )

                plt.close(density_figure)
                figure_paths[
                    f"{_safe_name(dataset)}"
                    "_automata_transition_density_heatmap"
                ] = density_heatmap_path

    if artifacts_dir.exists():
        for task_directory in sorted(artifacts_dir.iterdir()):
            if not task_directory.is_dir():
                continue

            transition_matrix_path = task_directory / "transition_matrix.csv"
            edge_table_path = task_directory / "observed_transition_edges.csv"

            if transition_matrix_path.exists():
                matrix = pd.read_csv(transition_matrix_path, index_col=0)

                heatmap_path = (
                    figures_dir
                    / f"{_safe_name(task_directory.name)}__transition_heatmap.png"
                )

                heatmap_figure, _ = plot_transition_heatmap(
                    transition_matrix=matrix,
                    title=f"{task_directory.name} — Transition Heatmap",
                    save_path=heatmap_path,
                    dpi=dpi
                )

                plt.close(heatmap_figure)
                figure_paths[
                    f"{task_directory.name}_transition_heatmap"
                ] = heatmap_path

            if edge_table_path.exists():
                edge_table = pd.read_csv(edge_table_path)

                graph_path = (
                    figures_dir
                    / f"{_safe_name(task_directory.name)}__state_graph.png"
                )

                graph_figure, _ = plot_automata_graph_from_edges(
                    edge_table=edge_table,
                    title=f"{task_directory.name} — Observed State Graph",
                    save_path=graph_path,
                    max_edges=automata_graph_max_edges,
                    dpi=dpi
                )

                plt.close(graph_figure)
                figure_paths[f"{task_directory.name}_state_graph"] = graph_path

    summary = {
        "source_results_directory": str(results_path),
        "result_row_count": int(len(results_table)),
        "notes": {
            "model_comparison": (
                "Original baseline task rows only are included."
            ),
            "robustness_degradation": (
                "Noise F1 values are paired with the same task's "
                "original F1 value."
            ),
            "pooled_prediction_curves": (
                "Curves pool repeated-run prediction artifacts. "
                "For automata, curve score equals anomaly score minus "
                "the calibrated threshold of its originating run."
            )
        },
        "tables": {
            name: str(path)
            for name, path in table_paths.items()
        },
        "figures": {
            name: str(path)
            for name, path in figure_paths.items()
        }
    }

    summary_path = report_path / "report_summary.json"
    _write_json(summary, summary_path)

    return {
        "report_summary": summary_path,
        "tables": table_paths,
        "figures": figure_paths
    }