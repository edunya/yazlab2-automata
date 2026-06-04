"""
Tests for paired statistical significance analysis.

These tests use synthetic completed-result tables only.
No training or dataset processing is executed.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.reporting.result_report import generate_result_report_package
from src.reporting.statistical_analysis import (
    build_paired_wilcoxon_summary
)


def make_significant_paired_results() -> pd.DataFrame:
    rows = []

    seeds = [42, 123, 2026, 7, 999]

    for fold in [1, 2]:
        for position, seed in enumerate(seeds):
            rows.extend([
                {
                    "task_type": "deep_learning_robustness",
                    "task_id": f"gru_{fold}_{seed}",
                    "dataset": "SKAB",
                    "model": "gru",
                    "scenario": "original",
                    "fold": fold,
                    "seed": seed,
                    "f1_score": 0.90 + position * 0.001
                },
                {
                    "task_type": "deep_learning_robustness",
                    "task_id": f"lstm_{fold}_{seed}",
                    "dataset": "SKAB",
                    "model": "lstm",
                    "scenario": "original",
                    "fold": fold,
                    "seed": seed,
                    "f1_score": 0.70 + position * 0.001
                }
            ])

    return pd.DataFrame(rows)


def test_paired_wilcoxon_detects_clear_matched_difference():
    results_table = make_significant_paired_results()

    summary = build_paired_wilcoxon_summary(
        results_table=results_table,
        metric="f1_score",
        compared_models=["lstm", "gru"],
        scenario="original",
        alpha=0.05
    )

    assert summary.shape[0] == 1

    row = summary.iloc[0]

    assert row["dataset"] == "SKAB"
    assert row["n_pairs"] == 10
    assert row["better_mean_model"] == "gru"
    assert row["p_value_adjusted"] < 0.05
    assert bool(row["significant_after_correction"]) is True


def test_paired_wilcoxon_handles_identical_zero_f1_scores():
    rows = []

    for seed in [42, 123, 2026, 7, 999]:
        rows.extend([
            {
                "task_type": "deep_learning_robustness",
                "task_id": f"lstm_{seed}",
                "dataset": "BATADAL",
                "model": "lstm",
                "scenario": "original",
                "fold": None,
                "seed": seed,
                "f1_score": 0.0
            },
            {
                "task_type": "deep_learning_robustness",
                "task_id": f"cnn1d_{seed}",
                "dataset": "BATADAL",
                "model": "cnn1d",
                "scenario": "original",
                "fold": None,
                "seed": seed,
                "f1_score": 0.0
            }
        ])

    summary = build_paired_wilcoxon_summary(
        results_table=pd.DataFrame(rows),
        metric="f1_score",
        compared_models=["lstm", "cnn1d"],
        scenario="original",
        alpha=0.05
    )

    assert summary.shape[0] == 1
    assert summary.iloc[0]["p_value"] == pytest.approx(1.0)
    assert summary.iloc[0]["p_value_adjusted"] == pytest.approx(1.0)
    assert bool(summary.iloc[0]["significant_after_correction"]) is False


def test_report_package_exports_statistical_significance_table(tmp_path: Path):
    results_dir = tmp_path / "final_results"
    report_dir = tmp_path / "report_package"

    results_dir.mkdir(parents=True, exist_ok=True)

    results_table = make_significant_paired_results()
    results_table.to_csv(results_dir / "results_raw.csv", index=False)

    output = generate_result_report_package(
        results_dir=results_dir,
        output_dir=report_dir,
        dpi=100,
        automata_graph_max_edges=10,
        statistical_metric="f1_score",
        statistical_models=["lstm", "gru"],
        statistical_scenario="original",
        statistical_alpha=0.05
    )

    statistical_path = (
        report_dir / "tables" / "statistical_significance_summary.csv"
    )

    assert statistical_path.exists()
    assert "statistical_significance_summary" in output["tables"]

    statistical_table = pd.read_csv(statistical_path)

    assert statistical_table.shape[0] == 1
    assert statistical_table.iloc[0]["better_mean_model"] == "gru"