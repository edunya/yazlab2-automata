"""
Tests for SKAB data loading pipeline.

These tests require local SKAB raw data under:
data/raw/SKAB/valve1
data/raw/SKAB/valve2
"""

from pathlib import Path

import pytest

from src.data.load_skab import (
    get_skab_feature_columns,
    load_skab_dataset,
    prepare_skab_features_target,
    summarize_skab_dataset
)
from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKAB_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "SKAB"


pytestmark = pytest.mark.skipif(
    not SKAB_RAW_DIR.exists(),
    reason="Local SKAB raw data is not available."
)


def test_load_skab_dataset_shape_and_columns():
    config = load_config("skab")
    df = load_skab_dataset(config)

    assert len(df) == 22472
    assert "datetime" in df.columns
    assert "anomaly" in df.columns
    assert "changepoint" in df.columns
    assert "source_group" in df.columns
    assert "source_file" in df.columns


def test_load_skab_dataset_source_tracking():
    config = load_config("skab")
    df = load_skab_dataset(config)

    assert sorted(df["source_group"].unique().tolist()) == ["valve1", "valve2"]
    assert df["source_file"].nunique() == 20

    assert "valve1/0.csv" in set(df["source_file"])
    assert "valve2/0.csv" in set(df["source_file"])


def test_load_skab_dataset_labels_and_missing_values():
    config = load_config("skab")
    df = load_skab_dataset(config)

    assert int(df["anomaly"].sum()) == 7826
    assert set(df["anomaly"].unique().tolist()) == {0, 1}
    assert int(df.isna().sum().sum()) == 0


def test_skab_feature_columns():
    config = load_config("skab")
    df = load_skab_dataset(config)

    feature_columns = get_skab_feature_columns(df, config)

    assert len(feature_columns) == 8

    assert "datetime" not in feature_columns
    assert "anomaly" not in feature_columns
    assert "changepoint" not in feature_columns
    assert "source_group" not in feature_columns
    assert "source_file" not in feature_columns

    expected_features = {
        "Accelerometer1RMS",
        "Accelerometer2RMS",
        "Current",
        "Pressure",
        "Temperature",
        "Thermocouple",
        "Voltage",
        "Volume Flow RateRMS"
    }

    assert set(feature_columns) == expected_features


def test_prepare_skab_features_target():
    config = load_config("skab")
    df = load_skab_dataset(config)

    X, y, groups = prepare_skab_features_target(df, config)

    assert X.shape == (22472, 8)
    assert y.shape[0] == 22472
    assert groups.shape[0] == 22472

    assert y.sum() == 7826
    assert groups.nunique() == 20


def test_summarize_skab_dataset():
    config = load_config("skab")
    df = load_skab_dataset(config)

    summary = summarize_skab_dataset(df, config)

    assert summary["row_count"] == 22472
    assert summary["feature_count"] == 8
    assert summary["anomaly_count"] == 7826
    assert summary["missing_values"] == 0
    assert summary["source_group_count"] == 2
    assert summary["source_file_count"] == 20