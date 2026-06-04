"""
BATADAL data loading pipeline.

This module loads Training Dataset 2, cleans column names,
parses the datetime column and converts attack labels into
binary anomaly detection targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def validate_batadal_columns(
    df: pd.DataFrame,
    target_column: str,
    time_columns: List[str],
    file_path: Path
) -> None:
    """
    Validate that required BATADAL columns exist.

    Parameters
    ----------
    df:
        Loaded BATADAL dataframe.

    target_column:
        Target column name, expected to be ATT_FLAG.

    time_columns:
        Time-related columns, expected to include DATETIME.

    file_path:
        Source file path used in error messages.
    """
    required_columns = {target_column, *time_columns}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: "
            f"{sorted(missing_columns)}"
        )


def convert_batadal_target_to_binary(
    target: pd.Series,
    normal_label: int,
    anomaly_label: int
) -> pd.Series:
    """
    Convert original BATADAL labels into binary anomaly labels.

    Mapping
    -------
    normal_label  -> 0
    anomaly_label -> 1
    """
    expected_labels = {normal_label, anomaly_label}
    observed_labels = set(target.dropna().unique().tolist())
    unexpected_labels = observed_labels.difference(expected_labels)

    if unexpected_labels:
        raise ValueError(
            "Unexpected BATADAL labels found: "
            f"{sorted(unexpected_labels)}. "
            f"Expected only {sorted(expected_labels)}."
        )

    binary_target = target.map({
        normal_label: 0,
        anomaly_label: 1
    })

    if binary_target.isna().any():
        raise ValueError("Target conversion produced missing binary labels.")

    return binary_target.astype(int)


def load_batadal_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and clean BATADAL Training Dataset 2.

    Processing steps
    ----------------
    - Read training_dataset_2.csv.
    - Strip spaces from column names.
    - Validate DATETIME and ATT_FLAG columns.
    - Parse DATETIME as pandas datetime.
    - Verify chronological row order.
    - Convert ATT_FLAG from {-999, 1} to {0, 1}.

    Parameters
    ----------
    config:
        Merged BATADAL configuration dictionary.

    Returns
    -------
    pandas.DataFrame
        Cleaned BATADAL dataframe.
    """
    dataset_config = config["dataset"]

    raw_data_dir = Path(dataset_config["raw_data_dir"])
    used_file = dataset_config["used_file"]
    separator = dataset_config.get("csv_separator", ",")

    target_column = dataset_config["target_column"]
    time_columns = dataset_config["time_columns"]
    datetime_format = dataset_config["datetime_format"]

    normal_label = dataset_config["normal_label"]
    anomaly_label = dataset_config["anomaly_label"]

    file_path = raw_data_dir / used_file

    if not file_path.exists():
        raise FileNotFoundError(f"BATADAL data file not found: {file_path}")

    df = pd.read_csv(file_path, sep=separator)

    if dataset_config.get("strip_column_names", True):
        df.columns = df.columns.str.strip()

    validate_batadal_columns(
        df=df,
        target_column=target_column,
        time_columns=time_columns,
        file_path=file_path
    )

    datetime_column = time_columns[0]

    df[datetime_column] = pd.to_datetime(
        df[datetime_column],
        format=datetime_format,
        errors="raise"
    )

    if not df[datetime_column].is_monotonic_increasing:
        raise ValueError(
            "BATADAL rows are not in chronological order. "
            "Time-ordered splitting would be unsafe."
        )

    df[target_column] = convert_batadal_target_to_binary(
        target=df[target_column],
        normal_label=normal_label,
        anomaly_label=anomaly_label
    )

    return df


def get_batadal_feature_columns(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> List[str]:
    """
    Return model feature columns for BATADAL.

    Excludes:
    - DATETIME
    - ATT_FLAG
    """
    dataset_config = config["dataset"]

    target_column = dataset_config["target_column"]
    time_columns = set(dataset_config["time_columns"])

    excluded_columns = time_columns.union({target_column})

    feature_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No BATADAL feature columns found.")

    non_numeric_columns = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "BATADAL feature columns must be numeric. "
            f"Non-numeric columns found: {non_numeric_columns}"
        )

    return feature_columns


def prepare_batadal_features_target(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare X, y and timestamps for BATADAL.

    Returns
    -------
    X:
        Numerical feature dataframe.

    y:
        Binary anomaly labels.

    timestamps:
        Datetime values retained for time-ordered splitting and reporting.
    """
    dataset_config = config["dataset"]

    target_column = dataset_config["target_column"]
    datetime_column = dataset_config["time_columns"][0]

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column not found: {datetime_column}")

    feature_columns = get_batadal_feature_columns(df, config)

    X = df[feature_columns].copy()
    y = df[target_column].astype(int).copy()
    timestamps = df[datetime_column].copy()

    return X, y, timestamps


def summarize_batadal_dataset(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a summary dictionary for cleaned BATADAL data.
    """
    dataset_config = config["dataset"]

    target_column = dataset_config["target_column"]
    datetime_column = dataset_config["time_columns"][0]
    feature_columns = get_batadal_feature_columns(df, config)

    anomaly_count = int(df[target_column].sum())
    row_count = int(len(df))

    summary = {
        "row_count": row_count,
        "column_count": int(df.shape[1]),
        "feature_count": int(len(feature_columns)),
        "normal_count": int((df[target_column] == 0).sum()),
        "anomaly_count": anomaly_count,
        "anomaly_ratio": float(anomaly_count / row_count),
        "missing_values": int(df.isna().sum().sum()),
        "datetime_start": str(df[datetime_column].min()),
        "datetime_end": str(df[datetime_column].max()),
        "is_time_ordered": bool(df[datetime_column].is_monotonic_increasing),
        "feature_columns": feature_columns
    }

    return summary