"""
Generate report-ready tables and figures from completed experiment results.

This script never trains models.

Example
-------
python scripts/generate_result_report.py ^
    --results-dir experiments\controlled_runs\final_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.reporting.result_report import generate_result_report_package
from src.utils.config_loader import load_config


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate report tables and figures from result artifacts."
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing results_raw.csv and optional artifacts/."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional report output directory."
    )

    return parser


def main() -> None:
    """
    Generate report package without running experiments.
    """
    args = build_parser().parse_args()

    config = load_config()

    statistical_config = config["reporting"]["statistical_analysis"]

    output = generate_result_report_package(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        dpi=int(config["visualization"]["dpi"]),
        automata_graph_max_edges=int(
            config["visualization"]["automata_graph_max_edges"]
        ),
        statistical_metric=statistical_config["metric"],
        statistical_models=statistical_config["compared_models"],
        statistical_scenario=statistical_config["scenario"],
        statistical_alpha=float(statistical_config["alpha"])
    )

    print(
        json.dumps(
            {
                "report_summary": str(output["report_summary"]),
                "table_count": len(output["tables"]),
                "figure_count": len(output["figures"])
            },
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()