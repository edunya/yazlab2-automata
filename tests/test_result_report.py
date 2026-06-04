"""
Tests for result reporting and figure-package generation.

These tests use synthetic exported result rows and artifacts only.
No model training is executed.
"""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.reporting.result_report import (
    build_automata_parameter_summary,
    build_original_model_summary,
    build_robustness_degradation_summary,
    build_unseen_analysis_summary,
    generate_result_report_package
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = PROJECT_ROOT / "scripts" / "generate_result_report.py"


def build_synthetic_export_directory(tmp_path: Path) -> Path:
    """
    Create a small result-export directory resembling final output.
    """
    results_dir = tmp_path / "final_results"
    artifacts_dir = results_dir / "artifacts"

    results_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "task_id": "deep_learning_robustness__BATADAL__gru__seed42",
            "task_type": "deep_learning_robustness",
            "dataset": "BATADAL",
            "model": "gru",
            "scenario": "original",
            "f1_score": 0.80,
            "accuracy": 0.90,
            "precision": 0.80,
            "recall": 0.80,
            "roc_auc": 0.90,
            "average_precision": 0.85,
            "training_seconds": 10.0,
            "inference_seconds": 1.0
        },
        {
            "task_id": "deep_learning_robustness__BATADAL__gru__seed42",
            "task_type": "deep_learning_robustness",
            "dataset": "BATADAL",
            "model": "gru",
            "scenario": "gaussian_noise_0.05",
            "f1_score": 0.70,
            "accuracy": 0.85,
            "precision": 0.70,
            "recall": 0.70,
            "roc_auc": 0.82,
            "average_precision": 0.74,
            "training_seconds": 0.0,
            "inference_seconds": 1.1
        },
        {
            "task_id": "automata_robustness__BATADAL",
            "task_type": "automata_robustness",
            "dataset": "BATADAL",
            "model": "automata",
            "scenario": "original",
            "f1_score": 0.75,
            "accuracy": 0.86,
            "precision": 0.75,
            "recall": 0.75,
            "roc_auc": 0.84,
            "average_precision": 0.78,
            "training_seconds": 2.0,
            "inference_seconds": 0.3,
            "unseen_involved_decisions": 2,
            "unseen_decision_ratio": 0.20,
            "unseen_state_occurrences": 2,
            "unseen_state_occurrence_ratio": 0.10
        },
        {
            "task_id": "automata_robustness__BATADAL",
            "task_type": "automata_robustness",
            "dataset": "BATADAL",
            "model": "automata",
            "scenario": "gaussian_noise_0.05",
            "f1_score": 0.60,
            "accuracy": 0.76,
            "precision": 0.60,
            "recall": 0.60,
            "roc_auc": 0.70,
            "average_precision": 0.65,
            "training_seconds": 0.0,
            "inference_seconds": 0.4,
            "unseen_involved_decisions": 4,
            "unseen_decision_ratio": 0.40,
            "unseen_state_occurrences": 5,
            "unseen_state_occurrence_ratio": 0.25
        },
        {
            "task_id": "automata_parameter_analysis__BATADAL__w3_a3",
            "task_type": "automata_parameter_analysis",
            "dataset": "BATADAL",
            "model": "automata",
            "scenario": "original",
            "window_size": 3,
            "alphabet_size": 3,
            "f1_score": 0.72,
            "accuracy": 0.84,
            "precision": 0.72,
            "recall": 0.72,
            "state_count": 8,
            "training_seconds": 1.0,
            "inference_seconds": 0.2
        },
        {
            "task_id": "automata_parameter_analysis__BATADAL__w4_a3",
            "task_type": "automata_parameter_analysis",
            "dataset": "BATADAL",
            "model": "automata",
            "scenario": "original",
            "window_size": 4,
            "alphabet_size": 3,
            "f1_score": 0.75,
            "accuracy": 0.86,
            "precision": 0.75,
            "recall": 0.75,
            "state_count": 12,
            "training_seconds": 2.0,
            "inference_seconds": 0.3
        }
    ]

    pd.DataFrame(rows).to_csv(
        results_dir / "results_raw.csv",
        index=False
    )

    dl_original_dir = (
        artifacts_dir
        / "deep_learning_robustness__BATADAL__gru__seed42"
        / "original"
    )
    dl_original_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "target_index": [1, 2, 3, 4],
        "y_true": [0, 0, 1, 1],
        "score": [0.10, 0.25, 0.75, 0.90],
        "threshold": [0.50, 0.50, 0.50, 0.50],
        "y_pred": [0, 0, 1, 1]
    }).to_csv(dl_original_dir / "predictions.csv", index=False)

    automata_task_dir = artifacts_dir / "automata_robustness__BATADAL"
    automata_original_dir = automata_task_dir / "original"
    automata_original_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "target_index": [1, 2, 3, 4],
        "y_true": [0, 0, 1, 1],
        "score": [0.20, 0.40, 1.30, 1.60],
        "threshold": [1.00, 1.00, 1.00, 1.00],
        "y_pred": [0, 0, 1, 1]
    }).to_csv(automata_original_dir / "predictions.csv", index=False)

    pd.DataFrame(
        [[0.80, 0.20], [0.30, 0.70]],
        index=["aa", "ab"],
        columns=["aa", "ab"]
    ).to_csv(automata_task_dir / "transition_matrix.csv")

    pd.DataFrame({
        "from_state": ["aa", "ab"],
        "to_state": ["ab", "aa"],
        "observed_count": [8, 6],
        "probability": [0.80, 0.70]
    }).to_csv(
        automata_task_dir / "observed_transition_edges.csv",
        index=False
    )

    return results_dir


def test_reporting_summary_tables_are_constructed_correctly(tmp_path):
    results_dir = build_synthetic_export_directory(tmp_path)
    results_table = pd.read_csv(results_dir / "results_raw.csv")

    original_summary = build_original_model_summary(results_table)
    robustness_summary = build_robustness_degradation_summary(results_table)
    parameter_summary = build_automata_parameter_summary(results_table)
    unseen_summary = build_unseen_analysis_summary(results_table)

    assert set(original_summary["model"].tolist()) == {
        "gru", "automata"
    }

    gru_drop = robustness_summary[
        robustness_summary["model"] == "gru"
    ]["f1_score_drop_mean"].iloc[0]

    assert gru_drop == pytest.approx(0.10)

    assert parameter_summary.shape[0] == 2
    assert set(parameter_summary["window_size"].tolist()) == {3, 4}

    assert unseen_summary.shape[0] == 2
    assert "unseen_decision_ratio_mean" in unseen_summary.columns


def test_report_package_generates_tables_and_figures(tmp_path):
    results_dir = build_synthetic_export_directory(tmp_path)
    report_dir = tmp_path / "report_package"

    output = generate_result_report_package(
        results_dir=results_dir,
        output_dir=report_dir,
        dpi=100,
        automata_graph_max_edges=10
    )

    assert output["report_summary"].exists()

    assert (
        report_dir / "tables" / "original_model_summary.csv"
    ).exists()

    assert (
        report_dir / "tables" / "robustness_degradation_summary.csv"
    ).exists()

    assert (
        report_dir / "tables" / "automata_parameter_summary.csv"
    ).exists()

    assert (
        report_dir / "tables" / "automata_unseen_summary.csv"
    ).exists()

    assert (
        report_dir / "figures" / "original_f1_model_comparison.png"
    ).exists()

    assert (
        report_dir
        / "figures"
        / "BATADAL__gru__original__confusion_matrix.png"
    ).exists()

    assert (
        report_dir
        / "figures"
        / "BATADAL__automata__original__roc_curve.png"
    ).exists()

    assert (
        report_dir
        / "figures"
        / "BATADAL__automata_parameter_heatmap.png"
    ).exists()

    assert (
        report_dir
        / "figures"
        / "automata_robustness__BATADAL__transition_heatmap.png"
    ).exists()

    assert (
        report_dir
        / "figures"
        / "automata_robustness__BATADAL__state_graph.png"
    ).exists()


def test_report_summary_records_reporting_notes(tmp_path):
    results_dir = build_synthetic_export_directory(tmp_path)

    output = generate_result_report_package(
        results_dir=results_dir,
        output_dir=tmp_path / "report_package",
        dpi=100
    )

    summary = json.loads(
        output["report_summary"].read_text(encoding="utf-8")
    )

    assert summary["result_row_count"] == 6
    assert "pooled_prediction_curves" in summary["notes"]
    assert len(summary["tables"]) >= 4
    assert len(summary["figures"]) >= 6


def test_report_script_runs_directly_on_existing_results(tmp_path):
    results_dir = build_synthetic_export_directory(tmp_path)
    report_dir = tmp_path / "script_report_package"

    completed_process = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(report_dir)
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False
    )

    assert completed_process.returncode == 0
    assert (report_dir / "report_summary.json").exists()
    assert '"table_count"' in completed_process.stdout
    assert '"figure_count"' in completed_process.stdout