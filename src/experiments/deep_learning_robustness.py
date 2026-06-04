"""
Gaussian-noise robustness orchestration for deep learning models.

Correct robustness protocol
---------------------------
1. Train one model using clean training and validation partitions.
2. Evaluate the fitted model on the clean original test partition.
3. Apply Gaussian noise only to raw test observations.
4. Transform noisy test observations using the already fitted
   training-only preprocessor.
5. Evaluate the same fitted model on each noisy test partition.

The model is never retrained for individual noise levels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.data.load_batadal import prepare_batadal_features_target
from src.data.load_skab import prepare_skab_features_target
from src.data.splitting import (
    DatasetSplit,
    create_batadal_time_split,
    create_skab_nested_splits
)
from src.data.windowing import SequenceWindowData, create_windows_for_split
from src.evaluation.metrics import (
    BinaryClassificationResult,
    evaluate_binary_scores,
    logits_to_probabilities
)
from src.evaluation.runtime import RuntimeRecord
from src.experiments.scenarios import (
    GaussianNoiseSetting,
    build_gaussian_noise_settings,
    inject_gaussian_noise_into_test_rows
)
from src.models.model_factory import build_model
from src.training.datasets import create_data_loaders
from src.training.run_experiment import prepare_deep_learning_partitions
from src.training.trainer import (
    TrainingResult,
    fit_model,
    predict_logits,
    resolve_device
)
from src.utils.seed import set_seed
from src.utils.timer import Timer


@dataclass(frozen=True)
class DeepLearningScenarioEvaluation:
    """
    Evaluation result for one clean or noisy test scenario.
    """

    scenario: str
    noise_level: Optional[float]
    clean_model_reused: bool
    evaluation_result: BinaryClassificationResult
    runtime_record: RuntimeRecord
    test_scores: np.ndarray
    test_labels: np.ndarray
    test_target_indices: np.ndarray

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible scenario summary.
        """
        return {
            "scenario": self.scenario,
            "noise_level": self.noise_level,
            "clean_model_reused": self.clean_model_reused,
            "runtime": self.runtime_record.as_dict(),
            "test_metrics": self.evaluation_result.as_dict()
        }


@dataclass
class DeepLearningRobustnessResult:
    """
    Complete Gaussian-noise robustness result for one fitted DL model.

    One result represents one:
    - dataset
    - model
    - seed
    - optional SKAB fold

    The model is trained once on clean data and evaluated under multiple
    test conditions.
    """

    dataset: str
    model_name: str
    seed: int
    fold: Optional[int]
    split_name: str
    split_sizes: Dict[str, int]
    clean_window_counts: Dict[str, int]
    preprocessing_summary: Dict[str, Any]
    training_result: TrainingResult
    scenario_results: Dict[str, DeepLearningScenarioEvaluation]

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible robustness summary.
        """
        return {
            "dataset": self.dataset,
            "model": self.model_name,
            "seed": int(self.seed),
            "fold": self.fold,
            "split_name": self.split_name,
            "split_sizes": self.split_sizes,
            "clean_window_counts": self.clean_window_counts,
            "preprocessing": self.preprocessing_summary,
            "training": asdict(self.training_result),
            "training_protocol": {
                "trained_on": "clean_training_partition",
                "validated_on": "clean_validation_partition",
                "noise_applied_to": "test_only",
                "retrain_for_noise_levels": False
            },
            "scenario_results": {
                scenario_name: scenario_result.summary()
                for scenario_name, scenario_result
                in self.scenario_results.items()
            }
        }


def _build_noisy_windowed_partitions(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    preprocessor,
    setting: GaussianNoiseSetting,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None
) -> Dict[str, SequenceWindowData]:
    """
    Create windowed partitions after altering only raw test features.

    Crucially, the fitted training preprocessor is reused. No new scaler
    fitting is permitted for noisy scenarios.
    """
    noisy_X = inject_gaussian_noise_into_test_rows(
        X=X,
        split=split,
        setting=setting
    )

    transformed_noisy_X = preprocessor.transform(noisy_X)

    window_config = config["windowing"]

    return create_windows_for_split(
        X=transformed_noisy_X,
        y=y,
        split=split,
        sequence_length=int(window_config["sequence_length"]),
        groups=groups,
        timestamps=timestamps,
        label_strategy=window_config["label_strategy"]
    )


def _evaluate_fitted_model_on_partition(
    model: torch.nn.Module,
    windowed_partitions: Dict[str, SequenceWindowData],
    config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    scenario: str,
    seed: int,
    fold: Optional[int],
    device: torch.device,
    training_seconds: float,
    noise_level: Optional[float]
) -> DeepLearningScenarioEvaluation:
    """
    Evaluate an already fitted model on one test scenario.
    """
    training_config = config["training"]

    data_loaders = create_data_loaders(
        windowed_partitions=windowed_partitions,
        batch_size=int(training_config["batch_size"]),
        seed=seed,
        device=device,
        num_workers=int(training_config["num_workers"])
    )

    with Timer(f"{scenario}_test_inference") as inference_timer:
        test_logits, test_labels = predict_logits(
            model=model,
            data_loader=data_loaders["test"],
            device=device
        )

    if inference_timer.elapsed is None:
        raise RuntimeError("Scenario inference runtime measurement failed.")

    test_probabilities = logits_to_probabilities(test_logits)

    evaluation_result = evaluate_binary_scores(
        y_true=test_labels.astype(int),
        scores=test_probabilities,
        threshold=float(
            config["evaluation"]["deep_learning_probability_threshold"]
        ),
        score_name="probability",
        zero_division=int(config["evaluation"]["zero_division"])
    )

    runtime_record = RuntimeRecord(
        dataset=dataset_name,
        model=model_name,
        scenario=scenario,
        seed=seed,
        fold=fold,
        training_seconds=float(training_seconds),
        inference_seconds=float(inference_timer.elapsed),
        sample_count=int(len(test_labels))
    )

    return DeepLearningScenarioEvaluation(
        scenario=scenario,
        noise_level=noise_level,
        clean_model_reused=True,
        evaluation_result=evaluation_result,
        runtime_record=runtime_record,
        test_scores=test_probabilities,
        test_labels=test_labels.astype(int),
        test_target_indices=windowed_partitions["test"].target_indices
    )


def run_deep_learning_gaussian_robustness_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    seed: int,
    fold: Optional[int] = None,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None,
    device: Optional[str | torch.device] = None
) -> DeepLearningRobustnessResult:
    """
    Train once on clean data and evaluate original plus noisy test scenarios.

    This function performs one controlled robustness experiment. It does not
    automatically execute multiple seeds, folds or models.
    """
    set_seed(seed)

    resolved_device = (
        torch.device(device)
        if device is not None
        else resolve_device(config)
    )

    clean_partitions, preprocessor = prepare_deep_learning_partitions(
        X=X,
        y=y,
        split=split,
        config=config,
        groups=groups,
        timestamps=timestamps
    )

    training_config = config["training"]

    clean_data_loaders = create_data_loaders(
        windowed_partitions=clean_partitions,
        batch_size=int(training_config["batch_size"]),
        seed=seed,
        device=resolved_device,
        num_workers=int(training_config["num_workers"])
    )

    model = build_model(
        model_name=model_name,
        input_size=int(X.shape[1]),
        config=config
    )

    training_result = fit_model(
        model=model,
        data_loaders=clean_data_loaders,
        train_targets=clean_partitions["train"].y,
        config=config,
        device=resolved_device
    )

    scenario_results: Dict[str, DeepLearningScenarioEvaluation] = {}

    scenario_results["original"] = _evaluate_fitted_model_on_partition(
        model=model,
        windowed_partitions=clean_partitions,
        config=config,
        dataset_name=dataset_name,
        model_name=model_name,
        scenario="original",
        seed=seed,
        fold=fold,
        device=resolved_device,
        training_seconds=training_result.training_time_seconds,
        noise_level=None
    )

    for setting in build_gaussian_noise_settings(config):
        noisy_partitions = _build_noisy_windowed_partitions(
            X=X,
            y=y,
            split=split,
            config=config,
            preprocessor=preprocessor,
            setting=setting,
            groups=groups,
            timestamps=timestamps
        )

        if not np.array_equal(
            noisy_partitions["test"].target_indices,
            clean_partitions["test"].target_indices
        ):
            raise RuntimeError(
                "Noise scenario changed test target-index alignment."
            )

        if not np.array_equal(
            noisy_partitions["test"].y,
            clean_partitions["test"].y
        ):
            raise RuntimeError(
                "Noise scenario changed test labels unexpectedly."
            )

        scenario_results[setting.scenario_name] = (
            _evaluate_fitted_model_on_partition(
                model=model,
                windowed_partitions=noisy_partitions,
                config=config,
                dataset_name=dataset_name,
                model_name=model_name,
                scenario=setting.scenario_name,
                seed=seed,
                fold=fold,
                device=resolved_device,
                training_seconds=0.0,
                noise_level=setting.level
            )
        )

    return DeepLearningRobustnessResult(
        dataset=dataset_name,
        model_name=model_name,
        seed=seed,
        fold=fold,
        split_name=split.split_name,
        split_sizes=split.sizes(),
        clean_window_counts={
            partition_name: len(window_data)
            for partition_name, window_data in clean_partitions.items()
        },
        preprocessing_summary=preprocessor.summary(),
        training_result=training_result,
        scenario_results=scenario_results
    )


def run_batadal_deep_learning_gaussian_robustness(
    df: pd.DataFrame,
    config: Dict[str, Any],
    model_name: str,
    seed: int,
    device: Optional[str | torch.device] = None
) -> DeepLearningRobustnessResult:
    """
    Run BATADAL clean-model Gaussian-noise robustness evaluation.
    """
    X, y, timestamps = prepare_batadal_features_target(df, config)
    split = create_batadal_time_split(df, config)

    return run_deep_learning_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="BATADAL",
        model_name=model_name,
        seed=seed,
        fold=None,
        groups=None,
        timestamps=timestamps,
        device=device
    )


def run_skab_deep_learning_gaussian_robustness_fold(
    df: pd.DataFrame,
    config: Dict[str, Any],
    model_name: str,
    seed: int,
    fold: int,
    device: Optional[str | torch.device] = None
) -> DeepLearningRobustnessResult:
    """
    Run one SKAB fold under clean-model Gaussian-noise evaluation.
    """
    X, y, groups = prepare_skab_features_target(df, config)
    splits = create_skab_nested_splits(df, config)

    if fold < 1 or fold > len(splits):
        raise ValueError(
            f"fold must be between 1 and {len(splits)}, found {fold}."
        )

    return run_deep_learning_gaussian_robustness_split(
        X=X,
        y=y,
        split=splits[fold - 1],
        config=config,
        dataset_name="SKAB",
        model_name=model_name,
        seed=seed,
        fold=fold,
        groups=groups,
        timestamps=None,
        device=device
    )