"""
Experiment logging utilities.

This module saves experiment parameters and metrics using
JSON and CSV files.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    CSV + JSON based experiment logger.

    Each experiment has its own folder:

    experiments/logs/experiment_name/
        params.json
        metrics.csv
        summary.json
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: Optional[str] = None
    ) -> None:
        self.log_dir = Path(log_dir)

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.params_path = self.experiment_dir / "params.json"
        self.metrics_path = self.experiment_dir / "metrics.csv"
        self.summary_path = self.experiment_dir / "summary.json"

    def save_params(self, params: Dict[str, Any]) -> None:
        """
        Save experiment parameters as JSON.
        """
        self._save_json(params, self.params_path)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        split: Optional[str] = None
    ) -> None:
        """
        Append metric values to metrics.csv.

        Parameters
        ----------
        metrics:
            Metric dictionary such as:
            {"accuracy": 0.91, "f1_score": 0.88}

        step:
            Epoch, fold or experiment step.

        split:
            train, validation or test.
        """
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_name": self.experiment_name,
            "step": step,
            "split": split
        }

        row.update(metrics)

        file_exists = self.metrics_path.exists()

        with self.metrics_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """
        Save final experiment summary as JSON.
        """
        self._save_json(summary, self.summary_path)

    def save_artifact_json(self, data: Dict[str, Any], filename: str) -> Path:
        """
        Save additional JSON artifact inside the experiment folder.
        """
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        path = self.experiment_dir / filename
        self._save_json(data, path)

        return path

    @staticmethod
    def _save_json(data: Dict[str, Any], path: str | Path) -> None:
        """
        Save dictionary as JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)


def append_global_summary(
    summary: Dict[str, Any],
    summary_csv_path: str | Path
) -> None:
    """
    Append one experiment summary row to a global CSV file.
    """
    summary_csv_path = Path(summary_csv_path)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = summary_csv_path.exists()

    with summary_csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary)