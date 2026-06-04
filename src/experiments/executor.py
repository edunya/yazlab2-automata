"""
Controlled real-data experiment execution.

This module connects the previously implemented experiment runners to the
final task plan while enforcing execution guards.

Safety rules
------------
- Benchmark execution requires explicit benchmark authorization.
- Full final execution requires the explicit phrase "tamamla".
- Building or exporting a plan never launches training.
- Reused default automata parameter rows do not trigger duplicate fitting.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


from pathlib import Path
import pandas as pd
import torch
import json

import numpy as np

from src.automata.run_experiment import run_automata_original_split
from src.data.load_batadal import prepare_batadal_features_target
from src.data.load_skab import prepare_skab_features_target
from src.data.splitting import (
    create_batadal_time_split,
    create_skab_nested_splits
)
from src.experiments.automata_robustness import (
    compute_automata_transition_structure,
    run_batadal_automata_gaussian_robustness,
    run_skab_automata_gaussian_robustness_fold
)
from src.experiments.batch_orchestration import (
    ExperimentPlan,
    ExperimentTask,
    export_flat_result_rows
)
from src.experiments.deep_learning_robustness import (
    run_batadal_deep_learning_gaussian_robustness,
    run_skab_deep_learning_gaussian_robustness_fold
)
from src.experiments.scenarios import (
    AutomataParameterSetting,
    build_config_for_automata_setting
)
from src.experiments.artifact_export import (
    export_automata_robustness_artifacts,
    export_deep_learning_robustness_artifacts
)


@dataclass(frozen=True)
class DatasetRegistry:
    """
    Real loaded dataframes required by controlled task execution.
    """

    skab: pd.DataFrame
    batadal: pd.DataFrame

    def get(self, dataset_name: str) -> pd.DataFrame:
        """
        Return the dataframe corresponding to a task dataset name.
        """
        normalized_name = dataset_name.upper()

        if normalized_name == "SKAB":
            return self.skab

        if normalized_name == "BATADAL":
            return self.batadal

        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_task_by_id(
    plan: ExperimentPlan,
    task_id: str
) -> ExperimentTask:
    """
    Find one planned task by its stable task identifier.
    """
    matches = [
        task
        for task in plan.tasks
        if task.task_id == task_id
    ]

    if not matches:
        raise ValueError(f"Task ID was not found in experiment plan: {task_id}")

    if len(matches) > 1:
        raise RuntimeError(f"Duplicate task ID found in experiment plan: {task_id}")

    return matches[0]


def get_configured_benchmark_tasks(
    plan: ExperimentPlan,
    base_config: Dict[str, Any]
) -> tuple[ExperimentTask, ...]:
    """
    Return explicitly configured lightweight benchmark tasks.
    """
    task_ids = base_config["execution"]["benchmark_task_ids"]

    if not task_ids:
        raise ValueError("At least one benchmark task ID must be configured.")

    tasks = tuple(
        get_task_by_id(plan, task_id)
        for task_id in task_ids
    )

    if any(not task.execute for task in tasks):
        raise ValueError("Benchmark tasks must require actual execution.")

    return tasks


def _validate_authorization(
    supplied_phrase: Optional[str],
    required_phrase: str,
    action_name: str
) -> None:
    """
    Validate one explicit user authorization phrase.
    """
    if supplied_phrase is None:
        raise PermissionError(
            f"{action_name} requires explicit authorization phrase: "
            f"'{required_phrase}'."
        )

    if supplied_phrase.strip().lower() != required_phrase.strip().lower():
        raise PermissionError(
            f"Invalid authorization for {action_name}. "
            f"Required phrase: '{required_phrase}'."
        )

def authorize_execution_mode(
    mode: str,
    base_config: Dict[str, Any],
    authorization_phrase: Optional[str]
) -> None:
    """
    Validate authorization before loading real datasets or running tasks.

    Supported modes
    ---------------
    - plan_only: never requires authorization and never executes training.
    - benchmark: requires configured benchmark authorization phrase.
    - full_run: requires configured final-run authorization phrase.
    """
    normalized_mode = mode.strip().lower()
    execution_config = base_config["execution"]

    if normalized_mode == "plan_only":
        return

    if normalized_mode == "benchmark":
        if execution_config["benchmark_requires_confirmation"]:
            _validate_authorization(
                supplied_phrase=authorization_phrase,
                required_phrase=execution_config[
                    "benchmark_confirmation_phrase"
                ],
                action_name="benchmark execution"
            )
        return

    if normalized_mode == "full_run":
        if execution_config["full_run_requires_confirmation"]:
            _validate_authorization(
                supplied_phrase=authorization_phrase,
                required_phrase=execution_config[
                    "full_run_confirmation_phrase"
                ],
                action_name="full experiment execution"
            )
        return

    raise ValueError(
        "Unsupported controlled execution mode. "
        "Expected 'plan_only', 'benchmark' or 'full_run'."
    )

def _add_task_metadata(
    row: Dict[str, Any],
    task: ExperimentTask
) -> Dict[str, Any]:
    """
    Add stable plan metadata to one flat result row.
    """
    enriched_row = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "dataset": task.dataset,
        "model": task.model,
        "fold": task.fold,
        "seed": task.seed,
        "reused_result": not task.execute,
        "reuse_source_task_id": task.reuse_source_task_id
    }

    if task.setting is not None:
        for setting_key in [
            "context_length",
            "window_size",
            "alphabet_size"
        ]:
            if setting_key in task.setting:
                enriched_row[setting_key] = task.setting[setting_key]

    enriched_row.update(row)

    return enriched_row


def _rows_from_deep_learning_robustness_result(
    task: ExperimentTask,
    result: Any
) -> list[Dict[str, Any]]:
    """
    Flatten original/noise deep learning scenario outputs.
    """
    rows = []

    for scenario_name, scenario_result in result.scenario_results.items():
        metrics = scenario_result.evaluation_result.as_dict()
        runtime = scenario_result.runtime_record.as_dict()

        row = {
            "scenario": scenario_name,
            "noise_level": scenario_result.noise_level,
            "clean_model_reused": scenario_result.clean_model_reused,
            **metrics,
            "training_seconds": runtime["training_seconds"],
            "inference_seconds": runtime["inference_seconds"]
        }

        rows.append(_add_task_metadata(row, task))

    return rows


def _rows_from_automata_robustness_result(
    task: ExperimentTask,
    result: Any
) -> list[Dict[str, Any]]:
    """
    Flatten original/noise automata outputs including unseen analysis.
    """
    rows = []

    transition_structure = compute_automata_transition_structure(
        result.fitted_pipeline.automata
    )

    for scenario_name, scenario_result in result.scenario_results.items():
        metrics = scenario_result.evaluation_result.as_dict()
        runtime = scenario_result.runtime_record.as_dict()
        unseen = scenario_result.unseen_summary.as_dict()

        row = {
            "scenario": scenario_name,
            **transition_structure,
            "noise_level": scenario_result.noise_level,
            "clean_pipeline_reused": scenario_result.clean_pipeline_reused,
            **metrics,
            "training_seconds": runtime["training_seconds"],
            "inference_seconds": runtime["inference_seconds"],
            "unseen_involved_decisions": unseen["unseen_involved_decisions"],
            "unseen_decision_ratio": unseen["unseen_decision_ratio"],
            "unseen_state_occurrences": unseen["unseen_state_occurrences"],
            "unseen_state_occurrence_ratio": unseen[
                "unseen_state_occurrence_ratio"
            ]
        }

        rows.append(_add_task_metadata(row, task))

    return rows


def _row_from_automata_parameter_result(
    task: ExperimentTask,
    result: Any
) -> Dict[str, Any]:
    """
    Flatten one executed automata parameter-analysis result.
    """
    metrics = result.evaluation_result.as_dict()
    runtime = result.runtime_record.as_dict()
    transition_structure = compute_automata_transition_structure(
        result.automata
    )

    row = {
        "scenario": "original",
        **metrics,
        "training_seconds": runtime["training_seconds"],
        "inference_seconds": runtime["inference_seconds"],
        **transition_structure,
        "threshold": result.calibration_result.threshold
    }

    return _add_task_metadata(row, task)


def _execute_automata_parameter_task(
    task: ExperimentTask,
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute one non-default automata parameter combination.
    """
    if task.setting is None:
        raise ValueError("Automata parameter task must include setting metadata.")

    setting = AutomataParameterSetting(
        window_size=int(task.setting["window_size"]),
        alphabet_size=int(task.setting["alphabet_size"]),
        context_length=int(task.setting["context_length"])
    )

    adjusted_config = build_config_for_automata_setting(
        config=config,
        setting=setting
    )

    if task.dataset == "BATADAL":
        X, y, timestamps = prepare_batadal_features_target(
            df,
            adjusted_config
        )
        split = create_batadal_time_split(df, adjusted_config)

        result = run_automata_original_split(
            X=X,
            y=y,
            split=split,
            config=adjusted_config,
            dataset_name="BATADAL",
            fold=None,
            groups=None,
            timestamps=timestamps,
            logger=None
        )

        return _row_from_automata_parameter_result(task, result)

    if task.dataset == "SKAB":
        if task.fold is None:
            raise ValueError("SKAB parameter task requires a fold value.")

        X, y, groups = prepare_skab_features_target(df, adjusted_config)
        splits = create_skab_nested_splits(df, adjusted_config)

        result = run_automata_original_split(
            X=X,
            y=y,
            split=splits[task.fold - 1],
            config=adjusted_config,
            dataset_name="SKAB",
            fold=task.fold,
            groups=groups,
            timestamps=None,
            logger=None
        )

        return _row_from_automata_parameter_result(task, result)

    raise ValueError(f"Unsupported task dataset: {task.dataset}")


def execute_executable_task(
    task: ExperimentTask,
    datasets: DatasetRegistry,
    configs_by_dataset: Mapping[str, Dict[str, Any]],
    device: Optional[str | torch.device] = None,
    artifacts_dir: Optional[str | Path] = None
) -> list[Dict[str, Any]]:
    """
    Execute one task that requires actual computation.

    This function does not itself authorize execution. It is called only
    after benchmark/full-run guard validation.
    """
    if not task.execute:
        raise ValueError(
            "A reused task cannot be executed. Materialize it from its "
            "baseline source result instead."
        )

    if task.dataset not in configs_by_dataset:
        raise ValueError(f"Missing config for dataset: {task.dataset}")

    config = configs_by_dataset[task.dataset]
    df = datasets.get(task.dataset)

    if task.task_type == "deep_learning_robustness":
        if task.seed is None:
            raise ValueError("Deep learning task requires a seed.")

        if task.dataset == "BATADAL":
            result = run_batadal_deep_learning_gaussian_robustness(
                df=df,
                config=config,
                model_name=task.model,
                seed=task.seed,
                device=device
            )
        elif task.dataset == "SKAB":
            if task.fold is None:
                raise ValueError("SKAB deep learning task requires a fold.")

            result = run_skab_deep_learning_gaussian_robustness_fold(
                df=df,
                config=config,
                model_name=task.model,
                seed=task.seed,
                fold=task.fold,
                device=device
            )
        else:
            raise ValueError(f"Unsupported task dataset: {task.dataset}")

        if artifacts_dir is not None:
            export_deep_learning_robustness_artifacts(
                task_id=task.task_id,
                result=result,
                output_dir=artifacts_dir
            )

        return _rows_from_deep_learning_robustness_result(task, result)

    if task.task_type == "automata_robustness":
        if task.dataset == "BATADAL":
            result = run_batadal_automata_gaussian_robustness(
                df=df,
                config=config
            )
        elif task.dataset == "SKAB":
            if task.fold is None:
                raise ValueError("SKAB automata task requires a fold.")

            result = run_skab_automata_gaussian_robustness_fold(
                df=df,
                config=config,
                fold=task.fold
            )
        else:
            raise ValueError(f"Unsupported task dataset: {task.dataset}")

        if artifacts_dir is not None:
            export_automata_robustness_artifacts(
                task_id=task.task_id,
                result=result,
                output_dir=artifacts_dir
            )

        return _rows_from_automata_robustness_result(task, result)

    if task.task_type == "automata_parameter_analysis":
        return [
            _execute_automata_parameter_task(
                task=task,
                df=df,
                config=config
            )
        ]

    raise ValueError(f"Unsupported task type: {task.task_type}")


def materialize_reused_task_rows(
    task: ExperimentTask,
    completed_rows_by_task: Mapping[str, list[Dict[str, Any]]]
) -> list[Dict[str, Any]]:
    """
    Materialize a reused default automata parameter result from baseline output.
    """
    if task.execute:
        raise ValueError("Only non-executable reuse tasks may be materialized.")

    if task.reuse_source_task_id is None:
        raise ValueError("Reused task must identify a source task.")

    if task.reuse_source_task_id not in completed_rows_by_task:
        raise ValueError(
            "Reuse source task has not been executed yet: "
            f"{task.reuse_source_task_id}"
        )

    source_rows = completed_rows_by_task[task.reuse_source_task_id]

    original_rows = [
        row
        for row in source_rows
        if row["scenario"] == "original"
    ]

    if len(original_rows) != 1:
        raise ValueError(
            "Reused automata parameter task requires exactly one "
            "original baseline row."
        )

    reused_row = deepcopy(original_rows[0])
    reused_row["task_id"] = task.task_id
    reused_row["task_type"] = task.task_type
    reused_row["reused_result"] = True
    reused_row["reuse_source_task_id"] = task.reuse_source_task_id

    if task.setting is not None:
        for setting_key in [
            "context_length",
            "window_size",
            "alphabet_size"
        ]:
            if setting_key in task.setting:
                reused_row[setting_key] = task.setting[setting_key]

    return [reused_row]

def _checkpoint_json_default(value: Any) -> Any:
    """
    Convert common NumPy values to JSON-compatible Python objects.
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def _task_checkpoint_path(
    checkpoint_dir: str | Path,
    task_id: str
) -> Path:
    """
    Return the checkpoint file path for one completed experiment task.
    """
    completed_tasks_dir = Path(checkpoint_dir) / "completed_tasks"
    completed_tasks_dir.mkdir(parents=True, exist_ok=True)

    return completed_tasks_dir / f"{task_id}.json"


def save_completed_task_checkpoint(
    task_id: str,
    task_rows: list[Dict[str, Any]],
    checkpoint_dir: str | Path
) -> Path:
    """
    Save completed result rows for one task immediately after completion.
    """
    if not task_rows:
        raise ValueError("A completed task checkpoint must contain result rows.")

    checkpoint_path = _task_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        task_id=task_id
    )

    with checkpoint_path.open("w", encoding="utf-8") as checkpoint_file:
        json.dump(
            task_rows,
            checkpoint_file,
            ensure_ascii=False,
            indent=2,
            default=_checkpoint_json_default
        )

    return checkpoint_path


def load_completed_task_checkpoint(
    task_id: str,
    checkpoint_dir: str | Path
) -> Optional[list[Dict[str, Any]]]:
    """
    Load saved task rows when a previous full run completed this task.

    Returns None when no checkpoint exists, meaning the task still needs
    actual execution.
    """
    checkpoint_path = _task_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        task_id=task_id
    )

    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open("r", encoding="utf-8") as checkpoint_file:
        task_rows = json.load(checkpoint_file)

    if not isinstance(task_rows, list) or not task_rows:
        raise ValueError(
            f"Invalid or empty task checkpoint: {checkpoint_path}"
        )

    return task_rows


def execute_benchmark_tasks(
    plan: ExperimentPlan,
    datasets: DatasetRegistry,
    configs_by_dataset: Mapping[str, Dict[str, Any]],
    base_config: Dict[str, Any],
    authorization_phrase: Optional[str],
    device: Optional[str | torch.device] = None,
    artifacts_dir: Optional[str | Path] = None
) -> list[Dict[str, Any]]:
    """
    Execute only configured lightweight benchmark tasks.

    This is still real computation and therefore requires explicit
    benchmark authorization.
    """
    authorize_execution_mode(
        mode="benchmark",
        base_config=base_config,
        authorization_phrase=authorization_phrase
    )

    rows = []

    for task in get_configured_benchmark_tasks(plan, base_config):
        rows.extend(
            execute_executable_task(
                task=task,
                datasets=datasets,
                configs_by_dataset=configs_by_dataset,
                device=device,
                artifacts_dir=artifacts_dir
            )
        )

    return rows


def execute_confirmed_full_plan(
    plan: ExperimentPlan,
    datasets: DatasetRegistry,
    configs_by_dataset: Mapping[str, Dict[str, Any]],
    base_config: Dict[str, Any],
    authorization_phrase: Optional[str],
    device: Optional[str | torch.device] = None,
    artifacts_dir: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
    resume: bool = True
) -> list[Dict[str, Any]]:
    """
    Execute the final plan only after explicit full-run authorization.
    """
    execution_config = base_config["execution"]

    authorize_execution_mode(
        mode="full_run",
        base_config=base_config,
        authorization_phrase=authorization_phrase
    )

    all_rows: list[Dict[str, Any]] = []
    completed_rows_by_task: Dict[str, list[Dict[str, Any]]] = {}

    checkpoint_path = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else None
    )

    for task in plan.tasks:
        task_rows: Optional[list[Dict[str, Any]]] = None

        if checkpoint_path is not None and resume:
            task_rows = load_completed_task_checkpoint(
                task_id=task.task_id,
                checkpoint_dir=checkpoint_path
            )

        if task_rows is None:
            if task.execute:
                task_rows = execute_executable_task(
                    task=task,
                    datasets=datasets,
                    configs_by_dataset=configs_by_dataset,
                    device=device,
                    artifacts_dir=artifacts_dir
                )
            else:
                task_rows = materialize_reused_task_rows(
                    task=task,
                    completed_rows_by_task=completed_rows_by_task
                )

            if checkpoint_path is not None:
                save_completed_task_checkpoint(
                    task_id=task.task_id,
                    task_rows=task_rows,
                    checkpoint_dir=checkpoint_path
                )

        completed_rows_by_task[task.task_id] = task_rows
        all_rows.extend(task_rows)

        if checkpoint_path is not None:
            export_flat_result_rows(
                result_rows=all_rows,
                output_dir=checkpoint_path / "partial_results"
            )

    return all_rows