"""
Detailed artifact export for completed experiment tasks.

These exports retain the information required for later reporting:
- sample-level labels, scores and predictions
- training history
- automata decision explanations
- automata transition matrix
- observed transition edge table

This module does not execute training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.evaluation.metrics import predictions_from_scores


def _json_default(value: Any) -> Any:
    """
    Convert common NumPy objects into JSON-compatible Python values.
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


def _write_json(data: Any, file_path: Path) -> None:
    """
    Write JSON content using UTF-8 and NumPy-safe serialization.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            data,
            json_file,
            ensure_ascii=False,
            indent=2,
            default=_json_default
        )


def _safe_name(value: str) -> str:
    """
    Convert task/scenario identifiers into filesystem-safe names.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _task_directory(
    output_dir: str | Path,
    task_id: str
) -> Path:
    """
    Create and return one task artifact directory.
    """
    task_dir = Path(output_dir) / _safe_name(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    return task_dir


def _build_prediction_table(
    target_indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> pd.DataFrame:
    """
    Build sample-level prediction table for one scenario.
    """
    if not (
        len(target_indices) == len(labels) == len(scores)
    ):
        raise ValueError(
            "target_indices, labels and scores must have identical lengths."
        )

    predictions = predictions_from_scores(
        scores=scores,
        threshold=threshold
    )

    return pd.DataFrame({
        "target_index": target_indices,
        "y_true": labels.astype(int),
        "score": scores.astype(float),
        "threshold": float(threshold),
        "y_pred": predictions.astype(int)
    })


def export_deep_learning_robustness_artifacts(
    task_id: str,
    result: Any,
    output_dir: str | Path
) -> Dict[str, Path]:
    """
    Export detailed artifacts from one deep learning robustness result.
    """
    task_dir = _task_directory(output_dir, task_id)

    summary_path = task_dir / "task_summary.json"
    history_path = task_dir / "training_history.json"

    _write_json(result.summary(), summary_path)
    _write_json(
        {"history": result.training_result.history},
        history_path
    )

    exported_paths: Dict[str, Path] = {
        "task_summary": summary_path,
        "training_history": history_path
    }

    for scenario_name, scenario_result in result.scenario_results.items():
        scenario_dir = task_dir / _safe_name(scenario_name)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        threshold = scenario_result.evaluation_result.threshold

        prediction_table = _build_prediction_table(
            target_indices=scenario_result.test_target_indices,
            labels=scenario_result.test_labels,
            scores=scenario_result.test_scores,
            threshold=threshold
        )

        predictions_path = scenario_dir / "predictions.csv"
        scenario_summary_path = scenario_dir / "summary.json"

        prediction_table.to_csv(predictions_path, index=False)
        _write_json(scenario_result.summary(), scenario_summary_path)

        exported_paths[f"{scenario_name}_predictions"] = predictions_path
        exported_paths[f"{scenario_name}_summary"] = scenario_summary_path

    return exported_paths


def _observed_transition_edges_table(automata: Any) -> pd.DataFrame:
    """
    Build table containing only transitions observed during training.

    Smoothed but unobserved transitions are not included.
    """
    rows = []

    for from_state, outgoing_counts in automata.transition_counts_.items():
        for to_state, count in outgoing_counts.items():
            rows.append({
                "from_state": from_state,
                "to_state": to_state,
                "observed_count": int(count),
                "probability": float(
                    automata.transition_probability(from_state, to_state)
                )
            })

    if not rows:
        raise ValueError(
            "No observed automata transitions are available for export."
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["probability", "observed_count"],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )


def export_automata_robustness_artifacts(
    task_id: str,
    result: Any,
    output_dir: str | Path
) -> Dict[str, Path]:
    """
    Export detailed artifacts from one automata robustness result.
    """
    task_dir = _task_directory(output_dir, task_id)

    summary_path = task_dir / "task_summary.json"
    pipeline_path = task_dir / "fitted_pipeline_summary.json"
    transition_matrix_path = task_dir / "transition_matrix.csv"
    observed_edges_path = task_dir / "observed_transition_edges.csv"

    _write_json(result.summary(), summary_path)
    _write_json(result.fitted_pipeline.summary(), pipeline_path)

    transition_matrix = (
        result.fitted_pipeline.automata.transition_matrix()
    )
    transition_matrix.to_csv(transition_matrix_path)

    observed_edges = _observed_transition_edges_table(
        result.fitted_pipeline.automata
    )
    observed_edges.to_csv(observed_edges_path, index=False)

    exported_paths: Dict[str, Path] = {
        "task_summary": summary_path,
        "fitted_pipeline_summary": pipeline_path,
        "transition_matrix": transition_matrix_path,
        "observed_transition_edges": observed_edges_path
    }

    for scenario_name, scenario_result in result.scenario_results.items():
        scenario_dir = task_dir / _safe_name(scenario_name)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        threshold = scenario_result.evaluation_result.threshold

        prediction_table = _build_prediction_table(
            target_indices=scenario_result.test_target_indices,
            labels=scenario_result.test_labels,
            scores=scenario_result.test_scores,
            threshold=threshold
        )

        predictions_path = scenario_dir / "predictions.csv"
        explanations_path = scenario_dir / "explanations.json"
        scenario_summary_path = scenario_dir / "summary.json"

        prediction_table.to_csv(predictions_path, index=False)

        _write_json(
            {
                "explanations": [
                    explanation.as_dict()
                    for explanation in scenario_result.test_explanations
                ]
            },
            explanations_path
        )

        _write_json(scenario_result.summary(), scenario_summary_path)

        exported_paths[f"{scenario_name}_predictions"] = predictions_path
        exported_paths[f"{scenario_name}_explanations"] = explanations_path
        exported_paths[f"{scenario_name}_summary"] = scenario_summary_path

    return exported_paths