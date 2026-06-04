"""
Tests for binary evaluation metrics and runtime reporting.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    evaluate_binary_scores,
    logits_to_probabilities,
    predictions_from_scores
)
from src.evaluation.runtime import (
    RuntimeRecord,
    runtime_records_to_table,
    summarize_runtime_table
)


def test_logits_to_probabilities():
    probabilities = logits_to_probabilities(
        np.asarray([-2.0, 0.0, 2.0])
    )

    assert probabilities[0] < 0.5
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] > 0.5


def test_predictions_from_scores_uses_high_score_as_anomaly():
    predictions = predictions_from_scores(
        scores=[0.10, 0.50, 0.90],
        threshold=0.50
    )

    assert predictions.tolist() == [0, 1, 1]


def test_evaluate_binary_scores_for_perfect_predictions():
    result = evaluate_binary_scores(
        y_true=[0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
        score_name="probability"
    )

    assert result.accuracy == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1_score == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.average_precision == pytest.approx(1.0)
    assert result.confusion_matrix == [[2, 0], [0, 2]]


def test_automata_anomaly_scores_use_same_high_score_convention():
    result = evaluate_binary_scores(
        y_true=[0, 1, 0, 1],
        scores=[0.2, 4.0, 0.4, 3.0],
        threshold=1.0,
        score_name="automata_anomaly_score"
    )

    assert result.f1_score == pytest.approx(1.0)
    assert result.predicted_anomaly_count == 2


def test_single_class_evaluation_has_no_auc_values():
    result = evaluate_binary_scores(
        y_true=[0, 0, 0],
        scores=[0.1, 0.2, 0.3],
        threshold=0.5
    )

    assert result.roc_auc is None
    assert result.average_precision is None
    assert result.f1_score == pytest.approx(0.0)


def test_runtime_table_and_summary():
    records = [
        RuntimeRecord(
            dataset="SKAB",
            model="lstm",
            scenario="original",
            seed=42,
            fold=1,
            training_seconds=10.0,
            inference_seconds=1.0,
            sample_count=100
        ),
        RuntimeRecord(
            dataset="SKAB",
            model="lstm",
            scenario="original",
            seed=123,
            fold=1,
            training_seconds=12.0,
            inference_seconds=1.4,
            sample_count=100
        )
    ]

    runtime_table = runtime_records_to_table(records)
    summary = summarize_runtime_table(runtime_table)

    assert runtime_table.shape[0] == 2
    assert summary.shape[0] == 1
    assert summary.iloc[0]["training_seconds_mean"] == pytest.approx(11.0)
    assert summary.iloc[0]["inference_seconds_mean"] == pytest.approx(1.2)
    assert summary.iloc[0]["run_count"] == 2