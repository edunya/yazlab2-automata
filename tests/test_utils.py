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

def test_automata_symbolic_config():
    config = load_config()

    automata_config = config["automata"]

    assert automata_config["context_length"] == 32
    assert automata_config["default_window_size"] == 4
    assert automata_config["default_alphabet_size"] == 3
    assert automata_config["window_size_values"] == [3, 4, 5, 6]
    assert automata_config["alphabet_size_values"] == [3, 4, 5, 6]

    assert (
        automata_config["state_construction"]
        == "sliding_context_paa_sax_word"
    )
    assert automata_config["state_label_strategy"] == "last_time_step"
    assert (
        automata_config["transition_target_strategy"]
        == "destination_state_label"
    )

    assert automata_config["sax_normalization"] == "train_zscore"
    assert automata_config["sax_breakpoint_strategy"] == "gaussian"

    assert automata_config["learning_strategy"] == "normal_train_runs"
    assert (
        automata_config["score_strategy"]
        == "mean_negative_log_probability"
    )
    assert automata_config["threshold_strategy"] == "validation_best_f1"
    assert (
        automata_config["threshold_tie_break"]
        == "higher_precision_then_higher_threshold"
    )

    assert (
        automata_config["unseen_mapping_strategy"]
        == "levenshtein_nearest_state"
    )
    assert automata_config["unseen_tie_break"] == "alphabetical_state"
    assert (
        automata_config["confidence_strategy"]
        == "relative_threshold_margin"
    )
    assert automata_config["confidence_is_probability"] is False
def test_evaluation_and_visualization_config():
    config = load_config()

    evaluation_config = config["evaluation"]
    visualization_config = config["visualization"]

    assert "f1_score" in evaluation_config["classification_metrics"]
    assert "roc_auc" in evaluation_config["classification_metrics"]
    assert "average_precision" in evaluation_config["classification_metrics"]

    assert evaluation_config["deep_learning_probability_threshold"] == 0.5
    assert (
        evaluation_config["score_positive_direction"]
        == "higher_score_is_anomaly"
    )

    assert visualization_config["figure_format"] == "png"
    assert visualization_config["dpi"] == 300
    assert visualization_config["automata_graph_max_edges"] == 30

def test_robustness_and_parameter_analysis_config():
    config = load_config()

    robustness_config = config["experiments"]["robustness"]
    noise_config = robustness_config["gaussian_noise"]
    unseen_config = robustness_config["unseen_analysis"]

    assert noise_config["levels"] == [0.01, 0.05, 0.10]
    assert noise_config["apply_to"] == "test_only"
    assert (
        noise_config["scale_reference"]
        == "training_feature_standard_deviation"
    )
    assert noise_config["retrain_model"] is False
    assert noise_config["noise_seed"] == 2026

    assert unseen_config["applies_to"] == ["automata"]
    assert unseen_config["report_seen_vs_unseen"] is True

    parameter_config = config["experiments"]["parameter_analysis"]["automata"]

    assert parameter_config["context_length_fixed"] == 32
    assert parameter_config["window_size_values"] == [3, 4, 5, 6]
    assert parameter_config["alphabet_size_values"] == [3, 4, 5, 6]
    assert parameter_config["primary_metric"] == "f1_score"

def test_execution_plan_config_requires_confirmation():
    config = load_config()

    execution_config = config["execution"]

    assert execution_config["full_run_requires_confirmation"] is True
    assert execution_config["benchmark_before_full_run"] is True
    assert execution_config["skab_fold_count"] == 5
    assert (
        execution_config["reuse_default_automata_parameter_result"]
        is True
    )

    default_setting = execution_config["default_automata_parameter_setting"]

    assert default_setting["window_size"] == 4
    assert default_setting["alphabet_size"] == 3
    assert execution_config["result_export_formats"] == ["csv", "json"]

def test_controlled_execution_guard_config():
    config = load_config()

    execution_config = config["execution"]

    assert execution_config["full_run_requires_confirmation"] is True
    assert execution_config["full_run_confirmation_phrase"] == "tamamla"

    assert execution_config["benchmark_before_full_run"] is True
    assert execution_config["benchmark_requires_confirmation"] is True
    assert execution_config["benchmark_confirmation_phrase"] == "benchmark"

    assert execution_config["benchmark_task_ids"] == [
        "deep_learning_robustness__BATADAL__gru__seed42"
    ]

    assert execution_config["resume_completed_tasks"] is True
    assert execution_config["checkpoint_results_after_each_task"] is True

def test_detailed_artifact_export_config():
    config = load_config()

    artifact_config = config["execution"]["export_artifacts"]

    assert artifact_config["save_sample_predictions"] is True
    assert artifact_config["save_automata_explanations"] is True
    assert artifact_config["save_automata_transition_tables"] is True

def test_reporting_config():
    config = load_config()

    reporting_config = config["reporting"]

    assert reporting_config["primary_metric"] == "f1_score"
    assert reporting_config["original_scenario"] == "original"

    assert reporting_config["baseline_task_types"] == [
        "deep_learning_robustness",
        "automata_robustness"
    ]

    assert (
        reporting_config["pool_prediction_curves_across_repeated_runs"]
        is True
    )

    assert (
        reporting_config["automata_curve_score"]
        == "score_minus_calibrated_threshold"
    )

    assert reporting_config["generate_transition_figures"] is True
    assert reporting_config["generate_parameter_heatmaps"] is True

    statistical_config = reporting_config["statistical_analysis"]

    assert statistical_config["method"] == "paired_wilcoxon"
    assert (
        statistical_config["multiple_comparison_correction"]
        == "holm_bonferroni"
    )
    assert statistical_config["metric"] == "f1_score"
    assert statistical_config["alpha"] == 0.05
    assert statistical_config["compared_models"] == [
        "lstm", "gru", "cnn1d"
    ]
    assert statistical_config["scenario"] == "original"

    assert reporting_config["generate_transition_density_heatmaps"] is True
    assert (
        reporting_config["transition_density_definition"]
        == "observed_directed_transitions_divided_by_state_count_squared"
    )