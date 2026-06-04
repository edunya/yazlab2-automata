from pathlib import Path
import json


PROJECT_DIRS = [
    "configs",

    "data/raw/SKAB",
    "data/raw/BATADAL",
    "data/interim",
    "data/processed",

    "src/data",
    "src/models",
    "src/automata",
    "src/training",
    "src/evaluation",
    "src/visualization",
    "src/utils",

    "experiments/logs",
    "experiments/metrics",
    "experiments/models",
    "experiments/predictions",

    "reports/figures",
    "reports/tables",

    "notebooks",
    "tests",
]


PYTHON_PACKAGE_DIRS = [
    "src",
    "src/data",
    "src/models",
    "src/automata",
    "src/training",
    "src/evaluation",
    "src/visualization",
    "src/utils",
]


PLACEHOLDER_FILES = [
    "data/raw/SKAB/.gitkeep",
    "data/raw/BATADAL/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",

    "experiments/logs/.gitkeep",
    "experiments/metrics/.gitkeep",
    "experiments/models/.gitkeep",
    "experiments/predictions/.gitkeep",

    "reports/figures/.gitkeep",
    "reports/tables/.gitkeep",

    "notebooks/.gitkeep",
]


PYTHON_FILES = [
    "src/data/load_skab.py",
    "src/data/load_batadal.py",
    "src/data/preprocessing.py",
    "src/data/splitting.py",
    "src/data/windowing.py",

    "src/models/lstm.py",
    "src/models/gru.py",
    "src/models/cnn1d.py",
    "src/models/model_factory.py",

    "src/automata/paa.py",
    "src/automata/sax.py",
    "src/automata/sliding_window.py",
    "src/automata/probabilistic_automata.py",
    "src/automata/levenshtein.py",
    "src/automata/explainability.py",

    "src/training/trainer.py",
    "src/training/early_stopping.py",
    "src/training/run_experiment.py",

    "src/evaluation/metrics.py",
    "src/evaluation/statistical_tests.py",
    "src/evaluation/robustness.py",

    "src/visualization/plot_metrics.py",
    "src/visualization/plot_confusion_matrix.py",
    "src/visualization/plot_roc_pr.py",
    "src/visualization/plot_automata_graph.py",
    "src/visualization/plot_transition_heatmap.py",

    "src/utils/config_loader.py",
    "src/utils/logger.py",
    "src/utils/seed.py",
    "src/utils/timer.py",

    "tests/test_paa.py",
    "tests/test_sax.py",
    "tests/test_levenshtein.py",
    "tests/test_automata_transitions.py",
]


BASE_CONFIG = {
    "project": {
        "name": "yazlab2-timeseries-automata",
        "version": "0.1.0"
    },
    "device": {
        "type": "auto"
    },
    "random_seeds": [42, 123, 2026, 7, 999],
    "training": {
        "max_epochs": 50,
        "batch_size": 32,
        "early_stopping_patience": 5,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "binary_cross_entropy"
    },
    "models": {
        "enabled_models": ["lstm", "gru", "cnn1d", "automata"],
        "default_sequence_length": 32
    },
    "automata": {
        "default_window_size": 4,
        "default_alphabet_size": 3,
        "window_size_values": [3, 4, 5, 6],
        "alphabet_size_values": [3, 4, 5, 6],
        "smoothing": 1e-8
    },
    "experiments": {
        "scenarios": ["original", "gaussian_noise", "unseen"],
        "main_metric": "f1_score",
        "metrics": ["accuracy", "precision", "recall", "f1_score"]
    },
    "paths": {
        "log_dir": "experiments/logs",
        "metrics_dir": "experiments/metrics",
        "models_dir": "experiments/models",
        "predictions_dir": "experiments/predictions",
        "figures_dir": "reports/figures",
        "tables_dir": "reports/tables"
    }
}


SKAB_CONFIG = {
    "dataset": {
        "name": "SKAB",
        "raw_data_dir": "data/raw/SKAB",
        "processed_data_dir": "data/processed/SKAB",
        "used_groups": ["valve1", "valve2"],
        "target_column": "anomaly",
        "group_column": "source_file",
        "excluded_columns": [
            "datetime",
            "changepoint",
            "source_group",
            "source_file"
        ],
        "split_strategy": "group_kfold",
        "n_splits": 5
    }
}


BATADAL_CONFIG = {
    "dataset": {
        "name": "BATADAL",
        "raw_data_dir": "data/raw/BATADAL",
        "processed_data_dir": "data/processed/BATADAL",
        "used_file": "Training Dataset 2",
        "target_column": None,
        "time_columns": [],
        "split_strategy": "time_ordered",
        "split_ratios": {
            "train": 0.60,
            "validation": 0.20,
            "test": 0.20
        }
    }
}


README_CONTENT = """# YazLab 2 - From Black-Box to Explainability

## Project Overview

This project compares deep learning based black-box models and probabilistic automata based interpretable models for time series anomaly detection.

## Datasets

- SKAB
- BATADAL

## Models

- LSTM
- GRU
- 1D-CNN
- Probabilistic Automata

## Automata Pipeline

- PCA
- PAA
- SAX
- Sliding Window
- State Transition Probability
- Levenshtein-based Unseen Pattern Handling

## Experimental Design

- Original data
- Gaussian noise scenario
- Unseen pattern scenario
- Parameter sensitivity analysis

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

## Visualizations

- Confusion Matrix
- ROC / Precision-Recall Curve
- Automata State Diagram
- Transition Probability Heatmap
- Parameter Sensitivity Plots

## How to Run

This section will be completed after implementation.

## Results

This section will be completed after experiments.

## Discussion

This section will be completed after analysis.

## Conclusion

This section will be completed after final evaluation.
"""


REQUIREMENTS_CONTENT = """numpy
pandas
scikit-learn
scipy
torch
matplotlib
seaborn
networkx
pytest
jupyter
"""


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = Path.cwd()

    for directory in PROJECT_DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)

    for package_dir in PYTHON_PACKAGE_DIRS:
        init_file = root / package_dir / "__init__.py"
        init_file.touch(exist_ok=True)

    for file_path in PLACEHOLDER_FILES:
        path = root / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    for file_path in PYTHON_FILES:
        path = root / file_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            path.write_text('"""Module placeholder."""\n', encoding="utf-8")

    write_json(root / "configs" / "base_config.json", BASE_CONFIG)
    write_json(root / "configs" / "skab_config.json", SKAB_CONFIG)
    write_json(root / "configs" / "batadal_config.json", BATADAL_CONFIG)

    write_text(root / "README.md", README_CONTENT)
    write_text(root / "requirements.txt", REQUIREMENTS_CONTENT)

    print("Project skeleton created successfully.")


if __name__ == "__main__":
    main()