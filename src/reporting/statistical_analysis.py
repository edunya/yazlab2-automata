"""
Statistical significance analysis for repeated deep learning experiments.

The project uses paired Wilcoxon signed-rank tests for deep learning
model comparisons because LSTM, GRU and 1D-CNN are evaluated on the
same original-scenario fold/seed combinations.

Automata is not included in this paired seed-based test because the
implemented probabilistic automata pipeline is deterministic and does
not generate seed-dependent repeated measurements.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


OUTPUT_COLUMNS = [
    "dataset",
    "scenario",
    "metric",
    "test_method",
    "correction_method",
    "model_a",
    "model_b",
    "n_pairs",
    "model_a_mean",
    "model_b_mean",
    "mean_difference_a_minus_b",
    "statistic",
    "p_value",
    "p_value_adjusted",
    "alpha",
    "significant_after_correction",
    "better_mean_model",
    "interpretation"
]


def _empty_result_table() -> pd.DataFrame:
    """
    Return an empty statistical-result table with stable columns.
    """
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _holm_bonferroni_adjust(p_values: np.ndarray) -> np.ndarray:
    """
    Apply Holm-Bonferroni multiple-comparison correction.
    """
    if len(p_values) == 0:
        return np.asarray([], dtype=np.float64)

    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)

    running_maximum = 0.0
    comparison_count = len(p_values)

    for rank, original_index in enumerate(order):
        adjusted_candidate = (
            (comparison_count - rank) * float(p_values[original_index])
        )

        running_maximum = max(running_maximum, adjusted_candidate)

        adjusted[original_index] = min(running_maximum, 1.0)

    return adjusted


def build_paired_wilcoxon_summary(
    results_table: pd.DataFrame,
    metric: str = "f1_score",
    compared_models: Sequence[str] = ("lstm", "gru", "cnn1d"),
    scenario: str = "original",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compare deep learning models using paired Wilcoxon signed-rank tests.

    Pairing rule
    ------------
    - BATADAL: same seed values are paired.
    - SKAB: same fold and same seed values are paired.

    Only deep_learning_robustness task rows are included.
    """
    required_columns = {
        "task_type",
        "dataset",
        "model",
        "scenario",
        "fold",
        "seed",
        metric
    }

    missing_columns = required_columns.difference(results_table.columns)

    if missing_columns:
        raise ValueError(
            "Results table is missing columns required for "
            f"statistical analysis: {sorted(missing_columns)}"
        )

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1.")

    model_names = list(compared_models)

    if len(model_names) < 2:
        raise ValueError("At least two models are required for comparison.")

    deep_learning_rows = results_table[
        (results_table["task_type"] == "deep_learning_robustness")
        & (results_table["scenario"] == scenario)
        & (results_table["model"].isin(model_names))
    ].copy()

    if deep_learning_rows.empty:
        return _empty_result_table()

    deep_learning_rows["_paired_fold"] = (
        deep_learning_rows["fold"].fillna(0).astype(int)
    )

    rows = []

    for dataset in sorted(deep_learning_rows["dataset"].unique().tolist()):
        dataset_rows = deep_learning_rows[
            deep_learning_rows["dataset"] == dataset
        ]

        pairing_columns = ["_paired_fold", "seed"]

        for model_a, model_b in combinations(model_names, 2):
            model_a_rows = dataset_rows[
                dataset_rows["model"] == model_a
            ][pairing_columns + [metric]].rename(
                columns={metric: "model_a_value"}
            )

            model_b_rows = dataset_rows[
                dataset_rows["model"] == model_b
            ][pairing_columns + [metric]].rename(
                columns={metric: "model_b_value"}
            )

            paired_values = model_a_rows.merge(
                model_b_rows,
                on=pairing_columns,
                how="inner",
                validate="one_to_one"
            )

            if paired_values.empty:
                continue

            values_a = paired_values["model_a_value"].to_numpy(
                dtype=np.float64
            )
            values_b = paired_values["model_b_value"].to_numpy(
                dtype=np.float64
            )

            differences = values_a - values_b

            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                test_result = wilcoxon(
                    values_a,
                    values_b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )

                statistic = float(test_result.statistic)
                p_value = float(test_result.pvalue)

            mean_a = float(values_a.mean())
            mean_b = float(values_b.mean())
            mean_difference = mean_a - mean_b

            if np.isclose(mean_difference, 0.0):
                better_mean_model = "tie"
            elif mean_difference > 0:
                better_mean_model = model_a
            else:
                better_mean_model = model_b

            rows.append({
                "dataset": dataset,
                "scenario": scenario,
                "metric": metric,
                "test_method": "paired_wilcoxon",
                "correction_method": "holm_bonferroni",
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": int(len(paired_values)),
                "model_a_mean": mean_a,
                "model_b_mean": mean_b,
                "mean_difference_a_minus_b": mean_difference,
                "statistic": statistic,
                "p_value": p_value,
                "alpha": float(alpha),
                "better_mean_model": better_mean_model
            })

    if not rows:
        return _empty_result_table()

    result_table = pd.DataFrame(rows)

    adjusted_parts = []

    for dataset, dataset_results in result_table.groupby(
        "dataset",
        sort=False
    ):
        adjusted_dataset_results = dataset_results.copy()

        adjusted_dataset_results["p_value_adjusted"] = (
            _holm_bonferroni_adjust(
                adjusted_dataset_results["p_value"].to_numpy(
                    dtype=np.float64
                )
            )
        )

        adjusted_parts.append(adjusted_dataset_results)

    result_table = pd.concat(adjusted_parts, ignore_index=True)

    result_table["significant_after_correction"] = (
        result_table["p_value_adjusted"] < result_table["alpha"]
    )

    interpretations = []

    for _, row in result_table.iterrows():
        if row["significant_after_correction"]:
            interpretations.append(
                f"Significant paired difference; higher mean "
                f"{metric}: {row['better_mean_model']}."
            )
        else:
            interpretations.append(
                "No statistically significant paired difference "
                "after Holm-Bonferroni correction."
            )

    result_table["interpretation"] = interpretations

    return result_table[OUTPUT_COLUMNS]