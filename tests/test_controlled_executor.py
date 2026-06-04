"""
Tests for controlled benchmark/full-run execution guards.

These tests do not start real model fitting.
"""

import pytest
import json

import src.experiments.executor as executor_module
from src.experiments.batch_orchestration import ExperimentPlan, ExperimentTask

from src.experiments.batch_orchestration import build_final_experiment_plan
from src.experiments.executor import (
    execute_benchmark_tasks,
    execute_confirmed_full_plan,
    get_configured_benchmark_tasks,
    materialize_reused_task_rows
)
from src.utils.config_loader import load_config


def test_configured_benchmark_task_is_lightweight_batadal_gru_task():
    config = load_config()
    plan = build_final_experiment_plan(config)

    benchmark_tasks = get_configured_benchmark_tasks(plan, config)

    assert len(benchmark_tasks) == 1

    task = benchmark_tasks[0]

    assert task.task_id == (
        "deep_learning_robustness__BATADAL__gru__seed42"
    )
    assert task.dataset == "BATADAL"
    assert task.model == "gru"
    assert task.seed == 42
    assert task.execute is True


def test_benchmark_cannot_execute_without_authorization():
    config = load_config()
    plan = build_final_experiment_plan(config)

    with pytest.raises(PermissionError):
        execute_benchmark_tasks(
            plan=plan,
            datasets=None,
            configs_by_dataset={},
            base_config=config,
            authorization_phrase=None,
            device="cpu"
        )


def test_full_run_cannot_execute_without_tamamla_authorization():
    config = load_config()
    plan = build_final_experiment_plan(config)

    with pytest.raises(PermissionError):
        execute_confirmed_full_plan(
            plan=plan,
            datasets=None,
            configs_by_dataset={},
            base_config=config,
            authorization_phrase="benchmark",
            device="cpu"
        )


def test_default_parameter_task_is_materialized_from_baseline_original_row():
    config = load_config()
    plan = build_final_experiment_plan(config)

    reused_task = next(
        task
        for task in plan.tasks
        if (
            task.task_type == "automata_parameter_analysis"
            and task.execute is False
            and task.dataset == "BATADAL"
        )
    )

    source_rows = {
        reused_task.reuse_source_task_id: [
            {
                "task_id": reused_task.reuse_source_task_id,
                "task_type": "automata_robustness",
                "dataset": "BATADAL",
                "model": "automata",
                "scenario": "original",
                "f1_score": 0.75,
                "reused_result": False
            },
            {
                "task_id": reused_task.reuse_source_task_id,
                "task_type": "automata_robustness",
                "dataset": "BATADAL",
                "model": "automata",
                "scenario": "gaussian_noise_0.05",
                "f1_score": 0.60,
                "reused_result": False
            }
        ]
    }

    materialized_rows = materialize_reused_task_rows(
        task=reused_task,
        completed_rows_by_task=source_rows
    )

    assert len(materialized_rows) == 1

    row = materialized_rows[0]

    assert row["scenario"] == "original"
    assert row["f1_score"] == 0.75
    assert row["task_id"] == reused_task.task_id
    assert row["task_type"] == "automata_parameter_analysis"
    assert row["reused_result"] is True
    assert row["window_size"] == 4
    assert row["alphabet_size"] == 3

def build_single_task_full_plan():
    return ExperimentPlan(
        tasks=(
            ExperimentTask(
                task_id="synthetic_completed_task",
                task_type="deep_learning_robustness",
                dataset="BATADAL",
                model="gru",
                scenarios=("original",),
                seed=42,
                execute=True
            ),
        ),
        confirmation_required=True,
        benchmark_required=True
    )


def test_full_execution_writes_task_checkpoint_and_partial_results(
    monkeypatch,
    tmp_path
):
    config = load_config()
    plan = build_single_task_full_plan()

    def fake_execute_executable_task(**kwargs):
        return [
            {
                "task_id": "synthetic_completed_task",
                "task_type": "deep_learning_robustness",
                "dataset": "BATADAL",
                "model": "gru",
                "scenario": "original",
                "f1_score": 0.75,
                "accuracy": 0.80
            }
        ]

    monkeypatch.setattr(
        executor_module,
        "execute_executable_task",
        fake_execute_executable_task
    )

    checkpoint_dir = tmp_path / "checkpoints"

    rows = executor_module.execute_confirmed_full_plan(
        plan=plan,
        datasets=None,
        configs_by_dataset={},
        base_config=config,
        authorization_phrase="tamamla",
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        resume=True
    )

    assert len(rows) == 1

    task_checkpoint = (
        checkpoint_dir
        / "completed_tasks"
        / "synthetic_completed_task.json"
    )

    assert task_checkpoint.exists()

    saved_rows = json.loads(
        task_checkpoint.read_text(encoding="utf-8")
    )

    assert saved_rows[0]["f1_score"] == 0.75

    assert (
        checkpoint_dir
        / "partial_results"
        / "results_raw.csv"
    ).exists()

    assert (
        checkpoint_dir
        / "partial_results"
        / "results_summary.csv"
    ).exists()


def test_full_execution_resume_skips_already_completed_task(
    monkeypatch,
    tmp_path
):
    config = load_config()
    plan = build_single_task_full_plan()
    checkpoint_dir = tmp_path / "checkpoints"

    def first_execution(**kwargs):
        return [
            {
                "task_id": "synthetic_completed_task",
                "task_type": "deep_learning_robustness",
                "dataset": "BATADAL",
                "model": "gru",
                "scenario": "original",
                "f1_score": 0.75,
                "accuracy": 0.80
            }
        ]

    monkeypatch.setattr(
        executor_module,
        "execute_executable_task",
        first_execution
    )

    first_rows = executor_module.execute_confirmed_full_plan(
        plan=plan,
        datasets=None,
        configs_by_dataset={},
        base_config=config,
        authorization_phrase="tamamla",
        checkpoint_dir=checkpoint_dir,
        resume=True
    )

    def execution_must_not_run_again(**kwargs):
        raise AssertionError("Completed task was unexpectedly re-executed.")

    monkeypatch.setattr(
        executor_module,
        "execute_executable_task",
        execution_must_not_run_again
    )

    resumed_rows = executor_module.execute_confirmed_full_plan(
        plan=plan,
        datasets=None,
        configs_by_dataset={},
        base_config=config,
        authorization_phrase="tamamla",
        checkpoint_dir=checkpoint_dir,
        resume=True
    )

    assert resumed_rows == first_rows