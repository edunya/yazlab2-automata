from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

SKAB_DIR = PROJECT_ROOT / "data" / "raw" / "SKAB"
BATADAL_DIR = PROJECT_ROOT / "data" / "raw" / "BATADAL"


def inspect_skab() -> None:
    print("\n" + "=" * 80)
    print("SKAB DATASET INSPECTION")
    print("=" * 80)

    valve_dirs = ["valve1", "valve2"]

    all_columns = {}
    total_rows = 0
    total_anomaly = 0
    total_missing = 0

    for valve in valve_dirs:
        valve_path = SKAB_DIR / valve
        csv_files = sorted(valve_path.glob("*.csv"))

        print(f"\n[{valve}]")
        print(f"Path: {valve_path}")
        print(f"CSV count: {len(csv_files)}")

        if not csv_files:
            print("WARNING: No CSV files found.")
            continue

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, sep=";")

            rows, cols = df.shape
            missing_count = int(df.isna().sum().sum())

            if "anomaly" in df.columns:
                anomaly_count = int(df["anomaly"].sum())
                anomaly_values = sorted(df["anomaly"].dropna().unique().tolist())
            else:
                anomaly_count = None
                anomaly_values = []

            has_datetime = "datetime" in df.columns
            has_changepoint = "changepoint" in df.columns
            has_anomaly = "anomaly" in df.columns

            all_columns[str(csv_file)] = list(df.columns)

            total_rows += rows
            total_missing += missing_count

            if anomaly_count is not None:
                total_anomaly += anomaly_count

            print(
                f"- {csv_file.name}: "
                f"rows={rows}, cols={cols}, "
                f"anomaly_count={anomaly_count}, "
                f"missing={missing_count}, "
                f"datetime={has_datetime}, "
                f"anomaly={has_anomaly}, "
                f"changepoint={has_changepoint}, "
                f"anomaly_values={anomaly_values}"
            )

    unique_column_sets = {
        tuple(columns)
        for columns in all_columns.values()
    }

    print("\n[SKAB SUMMARY]")
    print(f"Total rows: {total_rows}")
    print(f"Total anomaly rows: {total_anomaly}")
    print(f"Total missing values: {total_missing}")
    print(f"Unique column layouts: {len(unique_column_sets)}")

    if unique_column_sets:
        print("\nExample columns:")
        for column in list(unique_column_sets)[0]:
            print(f"  - {column}")

    if len(unique_column_sets) == 1:
        print("\nColumn consistency: OK")
    else:
        print("\nColumn consistency: WARNING - different column layouts detected")


def inspect_batadal() -> None:
    print("\n" + "=" * 80)
    print("BATADAL DATASET INSPECTION")
    print("=" * 80)

    csv_path = BATADAL_DIR / "training_dataset_2.csv"

    print(f"\nPath: {csv_path}")

    if not csv_path.exists():
        print("ERROR: training_dataset_2.csv not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print(f"Shape: {df.shape}")
    print(f"Missing values: {int(df.isna().sum().sum())}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    if "DATETIME" in df.columns:
        parsed_datetime = pd.to_datetime(
            df["DATETIME"],
            format="%d/%m/%y %H",
            errors="coerce"
        )

        print("\nDATETIME check:")
        print(f"  Parsed null count: {int(parsed_datetime.isna().sum())}")
        print(f"  Start: {parsed_datetime.min()}")
        print(f"  End: {parsed_datetime.max()}")

        diffs = parsed_datetime.diff().dropna()
        print(f"  Most common time diff: {diffs.mode().iloc[0] if not diffs.empty else None}")

    else:
        print("\nWARNING: DATETIME column not found.")

    if "ATT_FLAG" in df.columns:
        print("\nATT_FLAG distribution:")
        print(df["ATT_FLAG"].value_counts(dropna=False).sort_index())

        normal_count = int((df["ATT_FLAG"] == -999).sum())
        anomaly_count = int((df["ATT_FLAG"] == 1).sum())

        print("\nBinary interpretation:")
        print(f"  Normal rows ATT_FLAG=-999: {normal_count}")
        print(f"  Anomaly rows ATT_FLAG=1: {anomaly_count}")
        print(f"  Anomaly ratio: {anomaly_count / len(df):.4f}")

    else:
        print("\nWARNING: ATT_FLAG column not found.")

    feature_columns = [
        col for col in df.columns
        if col not in ["DATETIME", "ATT_FLAG"]
    ]

    print("\nFeature columns:")
    print(f"  Feature count: {len(feature_columns)}")
    for col in feature_columns:
        print(f"  - {col}")


def main() -> None:
    inspect_skab()
    inspect_batadal()


if __name__ == "__main__":
    main()