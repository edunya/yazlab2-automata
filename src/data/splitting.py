"""
Dataset splitting strategies.

This module implements:
- Nested group-safe splitting for SKAB.
- Time-ordered train/validation/test splitting for BATADAL.

Important leakage rule:
No preprocessing transformation is fitted here.
Scaling, PCA and later transformations must be fitted only on
the training indices produced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.load_batadal import prepare_batadal_features_target
from src.data.load_skab import prepare_skab_features_target


@dataclass
class DatasetSplit:
    """
    Stores train, validation and test row indices for one experiment split.
    """

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    split_name: str
    outer_fold: Optional[int] = None
    outer_method: Optional[str] = None
    validation_method: Optional[str] = None
    outer_fallback_reason: Optional[str] = None
    validation_fallback_reason: Optional[str] = None

    def sizes(self) -> Dict[str, int]:
        """
        Return partition sizes.
        """
        return {
            "train": int(len(self.train_indices)),
            "validation": int(len(self.validation_indices)),
            "test": int(len(self.test_indices))
        }


def _validate_group_count(
    groups: pd.Series,
    n_splits: int,
    split_label: str
) -> None:
    """
    Validate that enough unique groups exist for group-based splitting.
    """
    unique_group_count = int(groups.nunique())

    if unique_group_count < n_splits:
        raise ValueError(
            f"{split_label} requires at least {n_splits} unique groups, "
            f"but only {unique_group_count} were found."
        )


def _make_group_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    preferred_method: str,
    fallback_method: str,
    random_state: int,
    split_label: str
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str, Optional[str]]:
    """
    Create group-safe splits using the preferred strategy.

    StratifiedGroupKFold is attempted first when configured. If it cannot
    be used, GroupKFold is used as an explicit fallback.
    """
    _validate_group_count(groups, n_splits, split_label)

    if preferred_method == "stratified_group_kfold":
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )

            splits = list(splitter.split(X, y, groups))

            return splits, "stratified_group_kfold", None

        except (ImportError, ValueError) as error:
            if fallback_method != "group_kfold":
                raise

            fallback_reason = (
                f"{type(error).__name__}: {error}"
            )

            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(X, y, groups))

            return splits, "group_kfold_fallback", fallback_reason

    if preferred_method == "group_kfold":
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))

        return splits, "group_kfold", None

    raise ValueError(
        f"Unsupported group split strategy: {preferred_method}"
    )


def _validate_disjoint_indices(split: DatasetSplit) -> None:
    """
    Validate that train, validation and test row indices do not overlap.
    """
    train_set = set(split.train_indices.tolist())
    validation_set = set(split.validation_indices.tolist())
    test_set = set(split.test_indices.tolist())

    if train_set.intersection(validation_set):
        raise ValueError("Train and validation indices overlap.")

    if train_set.intersection(test_set):
        raise ValueError("Train and test indices overlap.")

    if validation_set.intersection(test_set):
        raise ValueError("Validation and test indices overlap.")


def _validate_group_isolation(
    split: DatasetSplit,
    groups: pd.Series
) -> None:
    """
    Validate that source_file groups are isolated across partitions.
    """
    train_groups = set(groups.iloc[split.train_indices].tolist())
    validation_groups = set(groups.iloc[split.validation_indices].tolist())
    test_groups = set(groups.iloc[split.test_indices].tolist())

    if train_groups.intersection(validation_groups):
        raise ValueError("A source_file appears in both train and validation.")

    if train_groups.intersection(test_groups):
        raise ValueError("A source_file appears in both train and test.")

    if validation_groups.intersection(test_groups):
        raise ValueError("A source_file appears in both validation and test.")


def create_skab_nested_splits(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> List[DatasetSplit]:
    """
    Create five nested, group-safe SKAB splits.

    Outer split
    -----------
    Separates the final test groups.

    Inner split
    -----------
    Separates validation groups only from the outer training/development pool.

    This prevents source_file leakage between train, validation and test sets.
    """
    dataset_config = config["dataset"]

    X, y, groups = prepare_skab_features_target(df, config)

    outer_strategy = dataset_config["split_strategy"]
    outer_fallback_strategy = dataset_config["fallback_split_strategy"]
    outer_n_splits = int(dataset_config["n_splits"])

    validation_strategy = dataset_config["validation_strategy"]
    validation_fallback_strategy = dataset_config[
        "validation_fallback_strategy"
    ]
    validation_n_splits = int(dataset_config["validation_n_splits"])

    split_seed = int(dataset_config["split_seed"])

    outer_splits, outer_method, outer_fallback_reason = _make_group_splits(
        X=X,
        y=y,
        groups=groups,
        n_splits=outer_n_splits,
        preferred_method=outer_strategy,
        fallback_method=outer_fallback_strategy,
        random_state=split_seed,
        split_label="SKAB outer split"
    )

    prepared_splits: List[DatasetSplit] = []

    for outer_fold, (development_indices, test_indices) in enumerate(
        outer_splits,
        start=1
    ):
        development_X = X.iloc[development_indices].reset_index(drop=True)
        development_y = y.iloc[development_indices].reset_index(drop=True)
        development_groups = groups.iloc[development_indices].reset_index(
            drop=True
        )

        (
            validation_splits,
            validation_method,
            validation_fallback_reason
        ) = _make_group_splits(
            X=development_X,
            y=development_y,
            groups=development_groups,
            n_splits=validation_n_splits,
            preferred_method=validation_strategy,
            fallback_method=validation_fallback_strategy,
            random_state=split_seed + outer_fold,
            split_label=f"SKAB validation split for outer fold {outer_fold}"
        )

        selected_validation_fold = (
            (outer_fold - 1) % len(validation_splits)
        )

        (
            train_relative_indices,
            validation_relative_indices
        ) = validation_splits[selected_validation_fold]

        train_indices = development_indices[train_relative_indices]
        validation_indices = development_indices[validation_relative_indices]

        split = DatasetSplit(
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            split_name="skab_nested_group_split",
            outer_fold=outer_fold,
            outer_method=outer_method,
            validation_method=validation_method,
            outer_fallback_reason=outer_fallback_reason,
            validation_fallback_reason=validation_fallback_reason
        )

        _validate_disjoint_indices(split)
        _validate_group_isolation(split, groups)

        prepared_splits.append(split)

    return prepared_splits


def create_batadal_time_split(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> DatasetSplit:
    """
    Create BATADAL time-ordered 60/20/20 train-validation-test split.

    Rows are never shuffled.
    """
    dataset_config = config["dataset"]

    X, y, timestamps = prepare_batadal_features_target(df, config)

    split_ratios = dataset_config["split_ratios"]

    train_ratio = float(split_ratios["train"])
    validation_ratio = float(split_ratios["validation"])
    test_ratio = float(split_ratios["test"])

    ratio_sum = train_ratio + validation_ratio + test_ratio

    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"BATADAL split ratios must sum to 1.0, found {ratio_sum}."
        )

    row_count = len(X)

    train_end = int(row_count * train_ratio)
    validation_end = train_end + int(row_count * validation_ratio)

    all_indices = np.arange(row_count)

    split = DatasetSplit(
        train_indices=all_indices[:train_end],
        validation_indices=all_indices[train_end:validation_end],
        test_indices=all_indices[validation_end:],
        split_name="batadal_time_ordered_split"
    )

    _validate_disjoint_indices(split)

    train_last_time = timestamps.iloc[split.train_indices[-1]]
    validation_first_time = timestamps.iloc[split.validation_indices[0]]
    validation_last_time = timestamps.iloc[split.validation_indices[-1]]
    test_first_time = timestamps.iloc[split.test_indices[0]]

    if train_last_time >= validation_first_time:
        raise ValueError("BATADAL train and validation time order is invalid.")

    if validation_last_time >= test_first_time:
        raise ValueError("BATADAL validation and test time order is invalid.")

    return split