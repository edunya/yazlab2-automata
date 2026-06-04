"""
Tests for sequence window generation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load_batadal import (
    load_batadal_dataset,
    prepare_batadal_features_target
)
from src.data.load_skab import (
    load_skab_dataset,
    prepare_skab_features_target
)
from src.data.preprocessing import DeepLearningPreprocessor
from src.data.splitting import (
    DatasetSplit,
    create_batadal_time_split,
    create_skab_nested_splits
)
from src.data.windowing import (
    create_sequence_windows,
    create_windows_for_split
)
from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKAB_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "SKAB"
BATADAL_RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "BATADAL" / "training_dataset_2.csv"
)


def test_sequence_window_uses_last_time_step_label():
    X = pd.DataFrame({
        "sensor": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    y = pd.Series([0, 1, 0, 1, 0])

    windows = create_sequence_windows(
        X=X,
        y=y,
        sequence_length=3,
        label_strategy="last_time_step"
    )

    assert windows.X.shape == (3, 3, 1)
    assert windows.y.tolist() == [0, 1, 0]
    assert windows.target_indices.tolist() == [2, 3, 4]


def test_sequence_windows_do_not_cross_group_boundaries():
    X = pd.DataFrame({
        "sensor": list(range(8))
    })
    y = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
    groups = pd.Series(["file_a"] * 4 + ["file_b"] * 4)

    windows = create_sequence_windows(
        X=X,
        y=y,
        sequence_length=3,
        groups=groups
    )

    assert len(windows) == 4
    assert windows.source_groups.tolist() == [
        "file_a", "file_a", "file_b", "file_b"
    ]
    assert windows.target_indices.tolist() == [2, 3, 6, 7]


def test_windows_are_created_after_partitioning():
    X = pd.DataFrame({
        "sensor": list(range(12))
    })
    y = pd.Series([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])

    split = DatasetSplit(
        train_indices=np.array([0, 1, 2, 3, 4]),
        validation_indices=np.array([5, 6, 7]),
        test_indices=np.array([8, 9, 10, 11]),
        split_name="synthetic_split"
    )

    partitions = create_windows_for_split(
        X=X,
        y=y,
        split=split,
        sequence_length=3
    )

    assert partitions["train"].target_indices.tolist() == [2, 3, 4]
    assert partitions["validation"].target_indices.tolist() == [7]
    assert partitions["test"].target_indices.tolist() == [10, 11]


def test_sequence_window_raises_when_partition_is_too_short():
    X = pd.DataFrame({"sensor": [1.0, 2.0]})
    y = pd.Series([0, 1])

    with pytest.raises(ValueError):
        create_sequence_windows(
            X=X,
            y=y,
            sequence_length=3
        )


@pytest.mark.skipif(
    not BATADAL_RAW_FILE.exists(),
    reason="Local BATADAL raw data is not available."
)
def test_batadal_window_counts_follow_time_partitions():
    config = load_config("batadal")
    df = load_batadal_dataset(config)

    X, y, timestamps = prepare_batadal_features_target(df, config)
    split = create_batadal_time_split(df, config)

    preprocessor = DeepLearningPreprocessor()
    preprocessor.fit(X.iloc[split.train_indices])
    transformed_X = preprocessor.transform(X)

    partitions = create_windows_for_split(
        X=transformed_X,
        y=y,
        split=split,
        sequence_length=32,
        timestamps=timestamps
    )

    assert len(partitions["train"]) == 2475
    assert len(partitions["validation"]) == 804
    assert len(partitions["test"]) == 805

    assert partitions["train"].X.shape[1:] == (32, 43)
    assert partitions["test"].target_timestamps is not None


@pytest.mark.skipif(
    not SKAB_RAW_DIR.exists(),
    reason="Local SKAB raw data is not available."
)
def test_skab_windows_respect_source_file_boundaries():
    config = load_config("skab")
    df = load_skab_dataset(config)

    X, y, groups = prepare_skab_features_target(df, config)
    split = create_skab_nested_splits(df, config)[0]

    preprocessor = DeepLearningPreprocessor()
    preprocessor.fit(X.iloc[split.train_indices])
    transformed_X = preprocessor.transform(X)

    partitions = create_windows_for_split(
        X=transformed_X,
        y=y,
        split=split,
        sequence_length=32,
        groups=groups
    )

    total_windows = (
        len(partitions["train"])
        + len(partitions["validation"])
        + len(partitions["test"])
    )

    assert total_windows == 21852

    assert partitions["train"].source_groups is not None
    assert partitions["validation"].source_groups is not None
    assert partitions["test"].source_groups is not None