"""
SKAB data loading pipeline.

This module loads only valve1 and valve2 folders, concatenates all CSV files,
and adds source tracking columns required for group-based splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "datetime",
    "anomaly",
    "changepoint"
}


def sort_csv_files(files: Iterable[Path]) -> List[Path]:
    """
    Sort CSV files numerically when possible.

    Example
    -------
    0.csv, 1.csv, 2.csv, ..., 10.csv
    """
    return sorted(
        files,
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem
    )


def validate_skab_columns(df: pd.DataFrame, file_path: Path) -> None:
    """
    Validate required SKAB columns.
    """
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: "
            f"{sorted(missing_columns)}"
        )


def load_skab_file(
    file_path: str | Path,
    source_group: str,
    separator: str = ";"
) -> pd.DataFrame:
    """
    Load one SKAB CSV file and add source tracking columns.

    Parameters
    ----------
    file_path:
        CSV file path.

    source_group:
        Folder/group name, for example valve1 or valve2.

    separator:
        CSV separator. SKAB files use semicolon.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe with source_group and source_file columns.
    """
    file_path = Path(file_path)

    df = pd.read_csv(file_path, sep=separator)
    df.columns = df.columns.str.strip()

    validate_skab_columns(df, file_path)

    df["source_group"] = source_group
    df["source_file"] = f"{source_group}/{file_path.name}"

    df["anomaly"] = df["anomaly"].astype(int)

    return df


def load_skab_dataset(config: Dict) -> pd.DataFrame:
    """
    Load and concatenate SKAB valve1 and valve2 CSV files.

    Parameters
    ----------
    config:
        Merged project configuration dictionary.

    Returns
    -------
    pandas.DataFrame
        Concatenated SKAB dataframe.
    """
    dataset_config = config["dataset"]

    raw_data_dir = Path(dataset_config["raw_data_dir"])
    used_groups = dataset_config["used_groups"]
    separator = dataset_config.get("csv_separator", ";")

    dataframes = []

    for source_group in used_groups:
        group_dir = raw_data_dir / source_group

        if not group_dir.exists():
            raise FileNotFoundError(f"SKAB group folder not found: {group_dir}")

        csv_files = sort_csv_files(group_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {group_dir}")

        for csv_file in csv_files:
            df_file = load_skab_file(
                file_path=csv_file,
                source_group=source_group,
                separator=separator
            )
            dataframes.append(df_file)

    if not dataframes:
        raise RuntimeError("No SKAB dataframes were loaded.")

    df = pd.concat(dataframes, ignore_index=True)

    return df


def get_skab_feature_columns(
    df: pd.DataFrame,
    config: Dict
) -> List[str]:
    """
    Return feature columns for SKAB.

    Excludes:
    - target column
    - datetime
    - changepoint
    - source_group
    - source_file
    """
    dataset_config = config["dataset"]

    target_column = dataset_config["target_column"]
    excluded_columns = set(dataset_config["excluded_columns"])
    excluded_columns.add(target_column)

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No SKAB feature columns found.")

    return feature_columns


def prepare_skab_features_target(
    df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare X, y and groups for SKAB.

    Returns
    -------
    X:
        Feature dataframe.

    y:
        Target labels.

    groups:
        source_file values for GroupKFold / StratifiedGroupKFold.
    """
    dataset_config = config["dataset"]

    target_column = dataset_config["target_column"]
    group_column = dataset_config["group_column"]

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")

    feature_columns = get_skab_feature_columns(df, config)

    X = df[feature_columns].copy()
    y = df[target_column].astype(int).copy()
    groups = df[group_column].copy()

    return X, y, groups


def summarize_skab_dataset(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Create a simple summary dictionary for SKAB.
    """
    target_column = config["dataset"]["target_column"]
    feature_columns = get_skab_feature_columns(df, config)

    summary = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "feature_count": int(len(feature_columns)),
        "anomaly_count": int(df[target_column].sum()),
        "normal_count": int((df[target_column] == 0).sum()),
        "missing_values": int(df.isna().sum().sum()),
        "source_group_count": int(df["source_group"].nunique()),
        "source_file_count": int(df["source_file"].nunique()),
        "source_groups": sorted(df["source_group"].unique().tolist()),
        "feature_columns": feature_columns
    }

    return summary