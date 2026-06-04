"""
Batch experiment planning and result-export orchestration.

This module prepares the final experiment manifest without launching
full training automatically.

Important execution rule
------------------------
Full real-data execution requires explicit user confirmation.
Until then, this module only:
- constructs the experiment plan,
- identifies reusable results,
- exports plan metadata,
- exports already produced flat metric rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import json

import numpy as np
import pandas as pd

from src.experiments.scenarios import create_automata_parameter_grid
from src.models.model_factory import DEEP_LEARNING_MODELS


@dataclass(frozen=True)
class ExperimentTask:
    """
    One planned experiment task.

    A task may be executable or may represent a result that will be reused
    from an already planned baseline task.
    """

    task_id: str
    task_type: str
    dataset: str
    model: str
    scenarios: tuple[str, ...]
    fold: Optional[int] = None
    seed: Optional[int] = None
    setting: Optional[Dict[str, Any]] = None
    execute: bool = True
    reuse_source_task_id: Optional[str] = None

    @property
    def expected_result_row_count(self) -> int:
        """
        Return the number of reporting rows produced by this task.
        """
        return len(self.scenarios)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-compatible task representation.
        """
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "model": self.model,
            "scenarios": list(self.scenarios),
            "fold": self.fold,
            "seed": self.seed,
            "setting": self.setting,
            "execute": self.execute,
            "reuse_source_task_id": self.reuse_source_task_id,
            "expected_result_row_count": self.expected_result_row_count
        }


@dataclass(frozen=True)
class ExperimentPlan:
    """
    Final experiment plan before actual execution.
    """

    tasks: tuple[ExperimentTask, ...]
    confirmation_required: bool
    benchmark_required: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def executable_tasks(self) -> tuple[ExperimentTask, ...]:
        """
        Return only tasks requiring actual computation.
        """
        return tuple(task for task in self.tasks if task.execute)

    def reused_tasks(self) -> tuple[ExperimentTask, ...]:
        """
        Return tasks represented through an already computed result.
        """
        return tuple(task for task in self.tasks if not task.execute)

    def expected_result_row_count(self) -> int:
        """
        Return number of rows expected in final flat result reporting.
        """
        return int(
            sum(task.expected_result_row_count for task in self.tasks)
        )

    def task_type_counts(self) -> Dict[str, int]:
        """
        Return plan-row counts by task type.
        """
        counts: Dict[str, int] = {}

        for task in self.tasks:
            counts[task.task_type] = counts.get(task.task_type, 0) + 1

        return counts

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible experiment plan summary.
        """
        return {
            "plan_row_count": len(self.tasks),
            "executable_task_count": len(self.executable_tasks()),
            "reused_task_count": len(self.reused_tasks()),
            "expected_result_row_count": self.expected_result_row_count(),
            "confirmation_required": self.confirmation_required,
            "benchmark_required": self.benchmark_required,
            "task_type_counts": self.task_type_counts(),
            "metadata": self.metadata
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return task manifest as a pandas DataFrame.
        """
        return pd.DataFrame([task.as_dict() for task in self.tasks])


def _fold_values(dataset: str, skab_fold_count: int) -> list[Optional[int]]:
    """
    Return required fold values for one dataset.
    """
    if dataset == "SKAB":
        return list(range(1, skab_fold_count + 1))

    if dataset == "BATADAL":
        return [None]

    raise ValueError(f"Unsupported dataset in execution plan: {dataset}")


def _baseline_automata_task_id(
    dataset: str,
    fold: Optional[int]
) -> str:
    """
    Return stable automata baseline task identifier.
    """
    fold_part = f"__fold{fold:02d}" if fold is not None else ""

    return f"automata_robustness__{dataset}{fold_part}"


def build_final_experiment_plan(config: Dict[str, Any]) -> ExperimentPlan:
    """
    Build the planned final experiment manifest.

    Planned execution protocol
    --------------------------
    Deep learning:
    - One clean training per model / seed / split.
    - Original plus configured Gaussian-noise test evaluations reuse the
      same fitted model.

    Automata robustness:
    - One clean fitted automata pipeline per dataset split.
    - Original plus configured Gaussian-noise test evaluations reuse it.
    - Unseen analysis is included in automata scenario output.

    Automata parameter analysis:
    - Every configured window/alphabet combination is reported.
    - The default combination is reused from baseline automata robustness,
      rather than fitted a second time.
    """
    execution_config = config["execution"]

    if not execution_config["full_run_requires_confirmation"]:
        raise ValueError(
            "Full final execution must require explicit confirmation."
        )

    skab_fold_count = int(execution_config["skab_fold_count"])

    enabled_dl_models = [
        model_name
        for model_name in config["models"]["enabled_models"]
        if model_name in DEEP_LEARNING_MODELS
    ]

    if not enabled_dl_models:
        raise ValueError("No deep learning models are enabled.")

    seeds = [int(seed) for seed in config["random_seeds"]]

    gaussian_settings = config["experiments"]["robustness"][
        "gaussian_noise"
    ]["levels"]

    robustness_scenarios = tuple(
        ["original"]
        + [
            f"gaussian_noise_{float(level):.2f}"
            for level in gaussian_settings
        ]
    )

    datasets = ["SKAB", "BATADAL"]
    tasks: list[ExperimentTask] = []

    # Deep learning clean-fit robustness tasks.
    for dataset in datasets:
        for fold in _fold_values(dataset, skab_fold_count):
            for model_name in enabled_dl_models:
                for seed in seeds:
                    fold_part = (
                        f"__fold{fold:02d}"
                        if fold is not None
                        else ""
                    )

                    task_id = (
                        f"deep_learning_robustness__{dataset}"
                        f"__{model_name}{fold_part}__seed{seed}"
                    )

                    tasks.append(
                        ExperimentTask(
                            task_id=task_id,
                            task_type="deep_learning_robustness",
                            dataset=dataset,
                            model=model_name,
                            scenarios=robustness_scenarios,
                            fold=fold,
                            seed=seed,
                            setting={
                                "trained_on": "clean_train",
                                "validated_on": "clean_validation",
                                "noise_applied_to": "test_only",
                                "retrain_for_noise": False
                            }
                        )
                    )

    # Automata original / Gaussian-noise / unseen analysis tasks.
    for dataset in datasets:
        for fold in _fold_values(dataset, skab_fold_count):
            task_id = _baseline_automata_task_id(dataset, fold)

            tasks.append(
                ExperimentTask(
                    task_id=task_id,
                    task_type="automata_robustness",
                    dataset=dataset,
                    model="automata",
                    scenarios=robustness_scenarios,
                    fold=fold,
                    seed=None,
                    setting={
                        "includes_unseen_analysis": True,
                        "refit_for_noise": False
                    }
                )
            )

    # Automata parameter-analysis tasks.
    parameter_settings = create_automata_parameter_grid(config)

    default_setting = execution_config[
        "default_automata_parameter_setting"
    ]
    default_window_size = int(default_setting["window_size"])
    default_alphabet_size = int(default_setting["alphabet_size"])

    reuse_default = bool(
        execution_config["reuse_default_automata_parameter_result"]
    )

    for dataset in datasets:
        for fold in _fold_values(dataset, skab_fold_count):
            baseline_task_id = _baseline_automata_task_id(dataset, fold)

            for setting in parameter_settings:
                is_default_setting = (
                    setting.window_size == default_window_size
                    and setting.alphabet_size == default_alphabet_size
                )

                fold_part = (
                    f"__fold{fold:02d}"
                    if fold is not None
                    else ""
                )

                task_id = (
                    f"automata_parameter_analysis__{dataset}"
                    f"{fold_part}__w{setting.window_size}"
                    f"_a{setting.alphabet_size}"
                )

                should_reuse = reuse_default and is_default_setting

                tasks.append(
                    ExperimentTask(
                        task_id=task_id,
                        task_type="automata_parameter_analysis",
                        dataset=dataset,
                        model="automata",
                        scenarios=("original",),
                        fold=fold,
                        seed=None,
                        setting=setting.as_dict(),
                        execute=not should_reuse,
                        reuse_source_task_id=(
                            baseline_task_id
                            if should_reuse
                            else None
                        )
                    )
                )

    plan = ExperimentPlan(
        tasks=tuple(tasks),
        confirmation_required=bool(
            execution_config["full_run_requires_confirmation"]
        ),
        benchmark_required=bool(
            execution_config["benchmark_before_full_run"]
        ),
        metadata={
            "datasets": datasets,
            "deep_learning_models": enabled_dl_models,
            "deep_learning_seeds": seeds,
            "skab_fold_count": skab_fold_count,
            "gaussian_scenarios": list(robustness_scenarios),
            "automata_is_deterministic": True,
            "unseen_analysis_is_part_of_automata_output": True
        }
    )

    return plan


def _json_default(value: Any) -> Any:
    """
    Convert common NumPy objects to JSON-compatible Python values.
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def export_experiment_plan(
    plan: ExperimentPlan,
    output_dir: str | Path
) -> Dict[str, Path]:
    """
    Export experiment plan manifest and summary without executing tasks.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_csv_path = output_path / "experiment_plan.csv"
    manifest_json_path = output_path / "experiment_plan.json"
    summary_json_path = output_path / "experiment_plan_summary.json"

    plan.to_dataframe().to_csv(manifest_csv_path, index=False)

    with manifest_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            [task.as_dict() for task in plan.tasks],
            json_file,
            ensure_ascii=False,
            indent=2,
            default=_json_default
        )

    with summary_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            plan.summary(),
            json_file,
            ensure_ascii=False,
            indent=2,
            default=_json_default
        )

    return {
        "manifest_csv": manifest_csv_path,
        "manifest_json": manifest_json_path,
        "summary_json": summary_json_path
    }


def summarize_flat_result_rows(results_table: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize already produced metric rows by dataset, model and scenario.

    This function does not execute experiments. It aggregates results that
    are supplied after execution.
    """
    required_columns = {
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

    available_metrics = [
        metric_name
        for metric_name in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
            "training_seconds",
            "inference_seconds"
        ]
        if metric_name in results_table.columns
    ]

    aggregations: Dict[str, tuple[str, str]] = {
        "run_count": ("model", "size")
    }

    for metric_name in available_metrics:
        aggregations[f"{metric_name}_mean"] = (metric_name, "mean")
        aggregations[f"{metric_name}_std"] = (metric_name, "std")

    summary = (
        results_table
        .groupby(["dataset", "model", "scenario"], as_index=False)
        .agg(**aggregations)
    )

    standard_deviation_columns = [
        column
        for column in summary.columns
        if column.endswith("_std")
    ]

    summary[standard_deviation_columns] = (
        summary[standard_deviation_columns].fillna(0.0)
    )

    return summary


def export_flat_result_rows(
    result_rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path
) -> Dict[str, Path]:
    """
    Export already generated flat metric results and grouped summaries.

    Full experiment execution will later supply the flat result rows.
    """
    rows = [dict(row) for row in result_rows]

    if not rows:
        raise ValueError("At least one result row is required for export.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_table = pd.DataFrame(rows)
    summary_table = summarize_flat_result_rows(results_table)

    raw_csv_path = output_path / "results_raw.csv"
    raw_json_path = output_path / "results_raw.json"
    summary_csv_path = output_path / "results_summary.csv"
    summary_json_path = output_path / "results_summary.json"

    results_table.to_csv(raw_csv_path, index=False)
    summary_table.to_csv(summary_csv_path, index=False)

    with raw_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            rows,
            json_file,
            ensure_ascii=False,
            indent=2,
            default=_json_default
        )

    with summary_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            summary_table.to_dict(orient="records"),
            json_file,
            ensure_ascii=False,
            indent=2,
            default=_json_default
        )

    return {
        "raw_csv": raw_csv_path,
        "raw_json": raw_json_path,
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path
    }