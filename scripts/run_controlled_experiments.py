"""
Controlled command-line entry point for project experiments.

Safe commands
-------------
Plan export only; runs no training:
    python scripts/run_controlled_experiments.py --plan-only

Benchmark; runs only configured lightweight task and requires authorization:
    python scripts/run_controlled_experiments.py --benchmark --authorization benchmark

Full final experiment execution; do not run until explicitly approved:
    python scripts/run_controlled_experiments.py --full-run --authorization tamamla
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.load_batadal import load_batadal_dataset
from src.data.load_skab import load_skab_dataset
from src.experiments.batch_orchestration import (
    build_final_experiment_plan,
    export_experiment_plan,
    export_flat_result_rows
)
from src.experiments.executor import (
    DatasetRegistry,
    execute_benchmark_tasks,
    execute_confirmed_full_plan
)
from src.utils.config_loader import load_config


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Controlled experiment runner for yazlab2 project."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        "--plan-only",
        action="store_true",
        help="Export experiment plan only; no training is executed."
    )

    mode_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Execute configured lightweight benchmark task."
    )

    mode_group.add_argument(
        "--full-run",
        action="store_true",
        help="Execute final full experiment plan."
    )

    parser.add_argument(
        "--authorization",
        type=str,
        default=None,
        help="Required confirmation phrase for benchmark or full run."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Optional device override."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "controlled_runs",
        help="Directory for plan and result exports."
    )

    return parser


def load_real_datasets_and_configs():
    """
    Load dataset-specific merged configs and raw datasets.
    """
    skab_config = load_config("skab")
    batadal_config = load_config("batadal")

    datasets = DatasetRegistry(
        skab=load_skab_dataset(skab_config),
        batadal=load_batadal_dataset(batadal_config)
    )

    configs_by_dataset = {
        "SKAB": skab_config,
        "BATADAL": batadal_config
    }

    return datasets, configs_by_dataset


def main() -> None:
    """
    Execute selected controlled runner mode.
    """
    args = build_parser().parse_args()

    base_config = load_config()
    plan = build_final_experiment_plan(base_config)

    plan_dir = args.output_dir / "plan"
    exported_plan_paths = export_experiment_plan(
        plan=plan,
        output_dir=plan_dir
    )

    print(json.dumps(plan.summary(), indent=2, ensure_ascii=False))
    print(f"Plan summary written to: {exported_plan_paths['summary_json']}")

    if args.plan_only:
        print("Plan-only mode completed. No training was executed.")
        return

    datasets, configs_by_dataset = load_real_datasets_and_configs()

    if args.benchmark:
        rows = execute_benchmark_tasks(
            plan=plan,
            datasets=datasets,
            configs_by_dataset=configs_by_dataset,
            base_config=base_config,
            authorization_phrase=args.authorization,
            device=args.device
        )

        output_dir = args.output_dir / "benchmark_results"

    else:
        rows = execute_confirmed_full_plan(
            plan=plan,
            datasets=datasets,
            configs_by_dataset=configs_by_dataset,
            base_config=base_config,
            authorization_phrase=args.authorization,
            device=args.device
        )

        output_dir = args.output_dir / "final_results"

    exported_result_paths = export_flat_result_rows(
        result_rows=rows,
        output_dir=output_dir
    )

    print(f"Result summary written to: {exported_result_paths['summary_json']}")


if __name__ == "__main__":
    main()