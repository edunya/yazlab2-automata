"""
Tests for validation-based automata threshold calibration.
"""

import numpy as np
import pytest

from src.automata.calibration import (
    calibrate_f1_threshold,
    predict_with_threshold
)


def test_predict_with_threshold_marks_high_scores_as_anomalies():
    scores = np.asarray([0.1, 0.5, 1.2])

    predictions = predict_with_threshold(
        scores=scores,
        threshold=0.5
    )

    assert predictions.tolist() == [0, 1, 1]


def test_calibration_finds_perfect_validation_threshold():
    validation_scores = np.asarray([0.10, 0.20, 1.00, 1.20])
    validation_labels = np.asarray([0, 0, 1, 1])

    result = calibrate_f1_threshold(
        validation_scores=validation_scores,
        validation_labels=validation_labels
    )

    assert result.threshold == pytest.approx(1.00)
    assert result.f1_score == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_calibration_summary_contains_reporting_fields():
    result = calibrate_f1_threshold(
        validation_scores=[0.1, 0.2, 1.0, 1.2],
        validation_labels=[0, 0, 1, 1]
    )

    summary = result.summary()

    assert summary["threshold_strategy"] == "validation_best_f1"
    assert summary["threshold_tie_break"] == (
        "higher_precision_then_higher_threshold"
    )
    assert summary["validation_f1_score"] == pytest.approx(1.0)


def test_calibration_requires_both_classes_in_validation():
    with pytest.raises(ValueError):
        calibrate_f1_threshold(
            validation_scores=[0.1, 0.2, 0.3],
            validation_labels=[0, 0, 0]
        )


def test_calibration_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calibrate_f1_threshold(
            validation_scores=[0.1, 0.2],
            validation_labels=[0]
        )