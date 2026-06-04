"""
Tests for final experiment planning and result export orchestration.

These tests build metadata only. They do not run real model training.
"""

import json

import pandas as pd
import pytest

from src.experiments.batch_orchestration import (
    build_final_experiment_plan,
    export_experiment_plan,
    export_flat_result_rows,
    summarize_flat_result_rows
)
from src.utils.config_loader import load_config


def test_final_plan_counts_expected_tasks_without_running_experiments():
    config = load_config()

    plan = build_final_experiment_plan(config)
    summary = plan.summary()

    assert summary["plan_row_count"] == 192
    assert summary["executable_task_count"] == 186
    assert summary["reused_task_count"] == 6
    assert summary["expected_result_row_count"] == 480

    assert summary["task_type_counts"] == {
        "deep_learning_robustness": 90,
        "automata_robustness": 6,
        "automata_parameter_analysis": 96
    }

    assert summary["confirmation_required"] is True
    assert summary["benchmark_required"] is True


def test_default_automata_parameter_rows_reuse_baseline_results():
    config = load_config()

    plan = build_final_experiment_plan(config)

    reused_parameter_tasks = [
        task
        for task in plan.tasks
        if (
            task.task_type == "automata_parameter_analysis"
            and not task.execute
        )
    ]

    assert len(reused_parameter_tasks) == 6

    for task in reused_parameter_tasks:
        assert task.setting["window_size"] == 4
        assert task.setting["alphabet_size"] == 3
        assert task.reuse_source_task_id is not None
        assert task.reuse_source_task_id.startswith("automata_robustness__")


def test_experiment_plan_can_be_exported_without_execution(tmp_path):
    config = load_config()

    plan = build_final_experiment_plan(config)

    exported_paths = export_experiment_plan(
        plan=plan,
        output_dir=tmp_path
    )

    assert exported_paths["manifest_csv"].exists()
    assert exported_paths["manifest_json"].exists()
    assert exported_paths["summary_json"].exists()

    summary = json.loads(
        exported_paths["summary_json"].read_text(encoding="utf-8")
    )

    manifest = pd.read_csv(exported_paths["manifest_csv"])

    assert summary["executable_task_count"] == 186
    assert manifest.shape[0] == 192


def test_flat_result_export_creates_raw_and_summary_tables(tmp_path):
    rows = [
        {
            "dataset": "SKAB",
            "model": "gru",
            "scenario": "original",
            "f1_score": 0.80,
            "accuracy": 0.90,
            "training_seconds": 10.0
        },
        {
            "dataset": "SKAB",
            "model": "gru",
            "scenario": "original",
            "f1_score": 0.84,
            "accuracy": 0.92,
            "training_seconds": 12.0
        }
    ]

    summary_table = summarize_flat_result_rows(pd.DataFrame(rows))

    assert summary_table.shape[0] == 1
    assert summary_table.iloc[0]["run_count"] == 2
    assert summary_table.iloc[0]["f1_score_mean"] == pytest.approx(0.82)
    assert summary_table.iloc[0]["training_seconds_mean"] == pytest.approx(11.0)

    exported_paths = export_flat_result_rows(
        result_rows=rows,
        output_dir=tmp_path
    )

    assert exported_paths["raw_csv"].exists()
    assert exported_paths["raw_json"].exists()
    assert exported_paths["summary_csv"].exists()
    assert exported_paths["summary_json"].exists()