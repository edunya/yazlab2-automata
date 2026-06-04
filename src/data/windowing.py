"""
Sequence window generation utilities.

The deep learning task is many-to-one sequence classification:
a sequence of observations is used to classify the label at its final
time step.

Important rules:
- SKAB windows must never cross source_file boundaries.
- BATADAL windows must be created after time-ordered partitioning.
- Windows must never cross train/validation/test boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.splitting import DatasetSplit


@dataclass
class SequenceWindowData:
    """
    Container for sequence classification windows.
    """

    X: np.ndarray
    y: np.ndarray
    target_indices: np.ndarray
    source_groups: Optional[np.ndarray] = None
    target_timestamps: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.y))

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Return model input shape: windows x sequence_length x feature_count.
        """
        return self.X.shape


def _validate_window_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    sequence_length: int,
    groups: Optional[pd.Series],
    timestamps: Optional[pd.Series],
    label_strategy: str
) -> None:
    """
    Validate input objects before window generation.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not X.index.equals(y.index):
        raise ValueError("X and y indices must be aligned.")

    if groups is not None:
        if len(groups) != len(X):
            raise ValueError("groups and X must contain the same number of rows.")

        if not X.index.equals(groups.index):
            raise ValueError("X and groups indices must be aligned.")

    if timestamps is not None:
        if len(timestamps) != len(X):
            raise ValueError(
                "timestamps and X must contain the same number of rows."
            )

        if not X.index.equals(timestamps.index):
            raise ValueError("X and timestamps indices must be aligned.")

    if label_strategy != "last_time_step":
        raise ValueError(
            "Unsupported label strategy. "
            "This project uses 'last_time_step'."
        )


def create_sequence_windows(
    X: pd.DataFrame,
    y: pd.Series,
    sequence_length: int,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None,
    label_strategy: str = "last_time_step"
) -> SequenceWindowData:
    """
    Create many-to-one sequence classification windows.

    Parameters
    ----------
    X:
        Feature dataframe.

    y:
        Row-level target labels.

    sequence_length:
        Number of observations in each sequence.

    groups:
        Optional group identifiers. When provided, windows are produced
        separately inside each group and never cross group boundaries.
        SKAB uses source_file values here.

    timestamps:
        Optional timestamps retained for prediction reporting.

    label_strategy:
        This project uses only "last_time_step".

    Returns
    -------
    SequenceWindowData
        Sequence input arrays and corresponding final-step labels.
    """
    _validate_window_inputs(
        X=X,
        y=y,
        sequence_length=sequence_length,
        groups=groups,
        timestamps=timestamps,
        label_strategy=label_strategy
    )

    feature_values = X.to_numpy(dtype=np.float32)

    sequences = []
    labels = []
    target_indices = []
    output_groups = []
    output_timestamps = []

    if groups is None:
        group_items = [(None, np.arange(len(X)))]
    else:
        group_items = [
            (
                group_value,
                np.flatnonzero(groups.to_numpy() == group_value)
            )
            for group_value in pd.unique(groups)
        ]

    for group_value, positions in group_items:
        if len(positions) < sequence_length:
            continue

        window_count = len(positions) - sequence_length + 1

        for start in range(window_count):
            window_positions = positions[start:start + sequence_length]
            target_position = window_positions[-1]

            sequences.append(feature_values[window_positions])
            labels.append(int(y.iloc[target_position]))
            target_indices.append(X.index[target_position])

            if groups is not None:
                output_groups.append(group_value)

            if timestamps is not None:
                output_timestamps.append(timestamps.iloc[target_position])

    if not sequences:
        raise ValueError(
            "No sequence windows could be created. "
            "Check partition size and sequence_length."
        )

    return SequenceWindowData(
        X=np.stack(sequences).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        target_indices=np.asarray(target_indices),
        source_groups=(
            np.asarray(output_groups, dtype=object)
            if groups is not None
            else None
        ),
        target_timestamps=(
            np.asarray(output_timestamps)
            if timestamps is not None
            else None
        )
    )


def create_windows_for_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    sequence_length: int,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None,
    label_strategy: str = "last_time_step"
) -> Dict[str, SequenceWindowData]:
    """
    Create separate train, validation and test windows.

    Partitioning is applied before window generation. Therefore,
    no window can cross train/validation/test boundaries.
    """
    partition_indices = {
        "train": split.train_indices,
        "validation": split.validation_indices,
        "test": split.test_indices
    }

    windowed_partitions: Dict[str, SequenceWindowData] = {}

    for partition_name, indices in partition_indices.items():
        partition_X = X.iloc[indices]
        partition_y = y.iloc[indices]

        partition_groups = (
            groups.iloc[indices]
            if groups is not None
            else None
        )

        partition_timestamps = (
            timestamps.iloc[indices]
            if timestamps is not None
            else None
        )

        windowed_partitions[partition_name] = create_sequence_windows(
            X=partition_X,
            y=partition_y,
            sequence_length=sequence_length,
            groups=partition_groups,
            timestamps=partition_timestamps,
            label_strategy=label_strategy
        )

    return windowed_partitions