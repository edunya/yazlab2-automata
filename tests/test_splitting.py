"""
Tests for dataset splitting strategies.

These tests require local SKAB and BATADAL raw data.
"""

from pathlib import Path

import numpy as np
import pytest

from src.data.load_batadal import load_batadal_dataset
from src.data.load_skab import load_skab_dataset
from src.data.splitting import (
    create_batadal_time_split,
    create_skab_nested_splits
)
from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKAB_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "SKAB"
BATADAL_RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "BATADAL" / "training_dataset_2.csv"
)


@pytest.mark.skipif(
    not SKAB_RAW_DIR.exists(),
    reason="Local SKAB raw data is not available."
)
def test_skab_nested_split_count_and_methods():
    config = load_config("skab")
    df = load_skab_dataset(config)

    splits = create_skab_nested_splits(df, config)

    assert len(splits) == 5

    for split in splits:
        assert split.outer_method in {
            "stratified_group_kfold",
            "group_kfold_fallback"
        }
        assert split.validation_method in {
            "stratified_group_kfold",
            "group_kfold_fallback"
        }


@pytest.mark.skipif(
    not SKAB_RAW_DIR.exists(),
    reason="Local SKAB raw data is not available."
)
def test_skab_splits_do_not_mix_source_files():
    config = load_config("skab")
    df = load_skab_dataset(config)

    splits = create_skab_nested_splits(df, config)

    for split in splits:
        train_groups = set(df.iloc[split.train_indices]["source_file"])
        validation_groups = set(df.iloc[split.validation_indices]["source_file"])
        test_groups = set(df.iloc[split.test_indices]["source_file"])

        assert train_groups.isdisjoint(validation_groups)
        assert train_groups.isdisjoint(test_groups)
        assert validation_groups.isdisjoint(test_groups)

        all_groups = train_groups | validation_groups | test_groups
        assert len(all_groups) == 20


@pytest.mark.skipif(
    not SKAB_RAW_DIR.exists(),
    reason="Local SKAB raw data is not available."
)
def test_skab_split_indices_are_complete_and_disjoint():
    config = load_config("skab")
    df = load_skab_dataset(config)

    splits = create_skab_nested_splits(df, config)

    expected_indices = set(range(len(df)))

    for split in splits:
        train_indices = set(split.train_indices.tolist())
        validation_indices = set(split.validation_indices.tolist())
        test_indices = set(split.test_indices.tolist())

        assert train_indices.isdisjoint(validation_indices)
        assert train_indices.isdisjoint(test_indices)
        assert validation_indices.isdisjoint(test_indices)

        combined_indices = train_indices | validation_indices | test_indices
        assert combined_indices == expected_indices


@pytest.mark.skipif(
    not SKAB_RAW_DIR.exists(),
    reason="Local SKAB raw data is not available."
)
def test_skab_splits_are_reproducible():
    config = load_config("skab")
    df = load_skab_dataset(config)

    first_splits = create_skab_nested_splits(df, config)
    second_splits = create_skab_nested_splits(df, config)

    for first, second in zip(first_splits, second_splits):
        assert np.array_equal(first.train_indices, second.train_indices)
        assert np.array_equal(
            first.validation_indices,
            second.validation_indices
        )
        assert np.array_equal(first.test_indices, second.test_indices)


@pytest.mark.skipif(
    not SKAB_RAW_DIR.exists(),
    reason="Local SKAB raw data is not available."
)
def test_skab_every_partition_contains_both_classes():
    config = load_config("skab")
    df = load_skab_dataset(config)

    splits = create_skab_nested_splits(df, config)

    for split in splits:
        train_labels = set(df.iloc[split.train_indices]["anomaly"].tolist())
        validation_labels = set(
            df.iloc[split.validation_indices]["anomaly"].tolist()
        )
        test_labels = set(df.iloc[split.test_indices]["anomaly"].tolist())

        assert train_labels == {0, 1}
        assert validation_labels == {0, 1}
        assert test_labels == {0, 1}


@pytest.mark.skipif(
    not BATADAL_RAW_FILE.exists(),
    reason="Local BATADAL raw data is not available."
)
def test_batadal_time_ordered_split_sizes():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    split = create_batadal_time_split(df, config)

    assert split.sizes() == {
        "train": 2506,
        "validation": 835,
        "test": 836
    }


@pytest.mark.skipif(
    not BATADAL_RAW_FILE.exists(),
    reason="Local BATADAL raw data is not available."
)
def test_batadal_time_order_is_preserved():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    split = create_batadal_time_split(df, config)

    train_times = df.iloc[split.train_indices]["DATETIME"]
    validation_times = df.iloc[split.validation_indices]["DATETIME"]
    test_times = df.iloc[split.test_indices]["DATETIME"]

    assert train_times.max() < validation_times.min()
    assert validation_times.max() < test_times.min()

    assert train_times.min().strftime("%Y-%m-%d %H:%M:%S") == (
        "2016-07-04 00:00:00"
    )
    assert test_times.max().strftime("%Y-%m-%d %H:%M:%S") == (
        "2016-12-25 00:00:00"
    )


@pytest.mark.skipif(
    not BATADAL_RAW_FILE.exists(),
    reason="Local BATADAL raw data is not available."
)
def test_batadal_anomaly_distribution_after_split():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    split = create_batadal_time_split(df, config)

    train_anomalies = int(df.iloc[split.train_indices]["ATT_FLAG"].sum())
    validation_anomalies = int(
        df.iloc[split.validation_indices]["ATT_FLAG"].sum()
    )
    test_anomalies = int(df.iloc[split.test_indices]["ATT_FLAG"].sum())

    assert train_anomalies == 102
    assert validation_anomalies == 37
    assert test_anomalies == 80
    assert train_anomalies + validation_anomalies + test_anomalies == 219