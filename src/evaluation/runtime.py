"""
Runtime reporting utilities.

This module stores and summarizes training and inference runtime values
for deep learning and probabilistic automata models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class RuntimeRecord:
    """
    Runtime record for one model evaluation run.
    """

    dataset: str
    model: str
    scenario: str
    seed: Optional[int]
    fold: Optional[int]
    training_seconds: float
    inference_seconds: float
    sample_count: int

    def as_dict(self) -> Dict[str, Any]:
        """
        Return dictionary representation.
        """
        return {
            "dataset": self.dataset,
            "model": self.model,
            "scenario": self.scenario,
            "seed": self.seed,
            "fold": self.fold,
            "training_seconds": float(self.training_seconds),
            "inference_seconds": float(self.inference_seconds),
            "sample_count": int(self.sample_count)
        }


def runtime_records_to_table(
    records: Iterable[RuntimeRecord]
) -> pd.DataFrame:
    """
    Convert runtime records into a pandas DataFrame.
    """
    record_list = list(records)

    if not record_list:
        raise ValueError("At least one runtime record is required.")

    return pd.DataFrame(
        [record.as_dict() for record in record_list]
    )


def summarize_runtime_table(
    runtime_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize runtime results by dataset, model and scenario.

    Returned columns include mean and standard deviation for:
    - training time
    - inference time
    """
    required_columns = {
        "dataset",
        "model",
        "scenario",
        "training_seconds",
        "inference_seconds"
    }

    missing_columns = required_columns.difference(runtime_table.columns)

    if missing_columns:
        raise ValueError(
            f"Runtime table is missing columns: {sorted(missing_columns)}"
        )

    summary = (
        runtime_table
        .groupby(["dataset", "model", "scenario"], as_index=False)
        .agg(
            training_seconds_mean=("training_seconds", "mean"),
            training_seconds_std=("training_seconds", "std"),
            inference_seconds_mean=("inference_seconds", "mean"),
            inference_seconds_std=("inference_seconds", "std"),
            run_count=("model", "size")
        )
    )

    standard_deviation_columns = [
        "training_seconds_std",
        "inference_seconds_std"
    ]

    summary[standard_deviation_columns] = (
        summary[standard_deviation_columns].fillna(0.0)
    )

    return summary