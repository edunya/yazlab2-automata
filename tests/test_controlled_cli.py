"""
Tests for the controlled command-line entry point.

These tests confirm:
- plan-only mode works when the script is executed directly,
- plan-only mode does not launch training,
- unauthorized benchmark requests are rejected before real execution.
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_controlled_experiments.py"


def test_plan_only_cli_runs_directly_without_training(tmp_path):
    output_dir = tmp_path / "controlled_plan_output"

    completed_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--plan-only",
            "--output-dir",
            str(output_dir)
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False
    )

    assert completed_process.returncode == 0
    assert "Plan-only mode completed. No training was executed." in (
        completed_process.stdout
    )

    assert (output_dir / "plan" / "experiment_plan.csv").exists()
    assert (output_dir / "plan" / "experiment_plan.json").exists()
    assert (output_dir / "plan" / "experiment_plan_summary.json").exists()


def test_benchmark_cli_rejects_missing_authorization_before_execution(tmp_path):
    output_dir = tmp_path / "unauthorized_benchmark_output"

    completed_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--benchmark",
            "--output-dir",
            str(output_dir)
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False
    )

    combined_output = (
        completed_process.stdout + completed_process.stderr
    )

    assert completed_process.returncode != 0
    assert "requires explicit authorization phrase" in combined_output
    assert not (output_dir / "benchmark_results").exists()