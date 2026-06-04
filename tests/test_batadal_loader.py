"""
Tests for BATADAL data loading pipeline.

These tests require local BATADAL raw data under:
data/raw/BATADAL/training_dataset_2.csv
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.load_batadal import (
    get_batadal_feature_columns,
    load_batadal_dataset,
    prepare_batadal_features_target,
    summarize_batadal_dataset
)
from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATADAL_RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "BATADAL" / "training_dataset_2.csv"
)


pytestmark = pytest.mark.skipif(
    not BATADAL_RAW_FILE.exists(),
    reason="Local BATADAL raw data is not available."
)


def test_load_batadal_dataset_shape_and_columns():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    assert df.shape == (4177, 45)
    assert "DATETIME" in df.columns
    assert "ATT_FLAG" in df.columns

    assert " L_T1" not in df.columns
    assert " ATT_FLAG" not in df.columns


def test_load_batadal_datetime_parsing_and_order():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    assert pd.api.types.is_datetime64_any_dtype(df["DATETIME"])
    assert df["DATETIME"].is_monotonic_increasing

    assert df["DATETIME"].min() == pd.Timestamp("2016-07-04 00:00:00")
    assert df["DATETIME"].max() == pd.Timestamp("2016-12-25 00:00:00")


def test_load_batadal_binary_target_and_missing_values():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    assert set(df["ATT_FLAG"].unique().tolist()) == {0, 1}
    assert int(df["ATT_FLAG"].sum()) == 219
    assert int((df["ATT_FLAG"] == 0).sum()) == 3958
    assert int(df.isna().sum().sum()) == 0


def test_batadal_feature_columns():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    feature_columns = get_batadal_feature_columns(df, config)

    assert len(feature_columns) == 43
    assert "DATETIME" not in feature_columns
    assert "ATT_FLAG" not in feature_columns

    expected_features = {
        "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7",
        "F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3",
        "F_PU4", "S_PU4", "F_PU5", "S_PU5", "F_PU6", "S_PU6",
        "F_PU7", "S_PU7", "F_PU8", "S_PU8", "F_PU9", "S_PU9",
        "F_PU10", "S_PU10", "F_PU11", "S_PU11",
        "F_V2", "S_V2",
        "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
        "P_J415", "P_J302", "P_J306", "P_J307", "P_J317",
        "P_J14", "P_J422"
    }

    assert set(feature_columns) == expected_features


def test_prepare_batadal_features_target():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    X, y, timestamps = prepare_batadal_features_target(df, config)

    assert X.shape == (4177, 43)
    assert y.shape[0] == 4177
    assert timestamps.shape[0] == 4177

    assert int(y.sum()) == 219
    assert timestamps.is_monotonic_increasing


def test_summarize_batadal_dataset():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    summary = summarize_batadal_dataset(df, config)

    assert summary["row_count"] == 4177
    assert summary["column_count"] == 45
    assert summary["feature_count"] == 43
    assert summary["normal_count"] == 3958
    assert summary["anomaly_count"] == 219
    assert summary["missing_values"] == 0
    assert summary["is_time_ordered"] is True
    assert summary["datetime_start"] == "2016-07-04 00:00:00"
    assert summary["datetime_end"] == "2016-12-25 00:00:00"