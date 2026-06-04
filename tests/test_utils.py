"""
Basic tests for utility modules.
"""

from pathlib import Path

import numpy as np

from src.utils.config_loader import get_value, load_config
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed
from src.utils.timer import Timer


def test_load_base_config():
    config = load_config()

    assert config["training"]["batch_size"] == 32
    assert config["training"]["max_epochs"] == 50
    assert config["random_seeds"] == [42, 123, 2026, 7, 999]


def test_load_skab_config():
    config = load_config("skab")

    assert config["dataset"]["name"] == "SKAB"
    assert config["dataset"]["target_column"] == "anomaly"
    assert config["dataset"]["split_strategy"] == "stratified_group_kfold"
    assert config["dataset"]["fallback_split_strategy"] == "group_kfold"
    assert config["dataset"]["n_splits"] == 5
    assert config["dataset"]["validation_n_splits"] == 4
    assert config["dataset"]["split_seed"] == 42


def test_load_batadal_config():
    config = load_config("batadal")

    assert config["dataset"]["name"] == "BATADAL"
    assert config["dataset"]["split_strategy"] == "time_ordered"
    assert config["dataset"]["split_ratios"]["train"] == 0.60


def test_get_value():
    config = load_config()

    batch_size = get_value(config, "training.batch_size")
    missing_value = get_value(config, "not.exists", default="missing")

    assert batch_size == 32
    assert missing_value == "missing"


def test_seed_reproducibility():
    set_seed(42)
    first = np.random.rand(5)

    set_seed(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_experiment_logger(tmp_path: Path):
    logger = ExperimentLogger(
        log_dir=tmp_path,
        experiment_name="test_experiment"
    )

    logger.save_params({"model": "lstm", "seed": 42})

    logger.log_metrics(
        metrics={"accuracy": 0.90, "f1_score": 0.88},
        step=1,
        split="validation"
    )

    logger.save_summary({"best_f1_score": 0.88})

    assert (tmp_path / "test_experiment" / "params.json").exists()
    assert (tmp_path / "test_experiment" / "metrics.csv").exists()
    assert (tmp_path / "test_experiment" / "summary.json").exists()


def test_timer():
    with Timer("test_timer") as timer:
        total = sum(range(100))

    assert total == 4950
    assert timer.elapsed is not None
    assert timer.elapsed >= 0

def test_preprocessing_and_windowing_config():
    config = load_config()

    assert config["preprocessing"]["normalization"] == "standard_scaler"
    assert config["preprocessing"]["automata_dimension_reduction"] == "pca"
    assert config["preprocessing"]["automata_n_components"] == 1

    assert config["windowing"]["sequence_length"] == 32
    assert config["windowing"]["label_strategy"] == "last_time_step"

def test_model_architecture_config():
    config = load_config()

    lstm_config = config["models"]["architectures"]["lstm"]
    gru_config = config["models"]["architectures"]["gru"]
    cnn_config = config["models"]["architectures"]["cnn1d"]

    assert lstm_config["hidden_size"] == 64
    assert lstm_config["num_layers"] == 1
    assert lstm_config["dropout"] == 0.2

    assert gru_config["hidden_size"] == 64
    assert gru_config["num_layers"] == 1
    assert gru_config["dropout"] == 0.2

    assert cnn_config["conv_channels"] == [32, 64]
    assert cnn_config["kernel_size"] == 3
    assert cnn_config["dropout"] == 0.2

def test_training_pipeline_config():
    config = load_config()

    training_config = config["training"]

    assert training_config["loss_function"] == "bce_with_logits"
    assert training_config["class_imbalance_strategy"] == "train_pos_weight"
    assert training_config["batch_size"] == 32
    assert training_config["max_epochs"] == 50
    assert training_config["early_stopping_patience"] == 5
    assert training_config["num_workers"] == 0
    assert training_config["use_amp"] is True