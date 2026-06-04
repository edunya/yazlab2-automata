"""
Original-scenario experiment orchestration for deep learning models.

This module connects the already tested components:

dataset features/labels -> split -> train-only preprocessing -> windowing
-> PyTorch model -> training/early stopping -> test probabilities
-> metrics -> runtime/loggable result.

The runner does not execute a full multi-seed experiment automatically.
It runs one model on one supplied split so that later batch orchestration
can remain explicit and controlled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.data.load_batadal import prepare_batadal_features_target
from src.data.load_skab import prepare_skab_features_target
from src.data.preprocessing import DeepLearningPreprocessor
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
from src.models.model_factory import build_model
from src.training.datasets import create_data_loaders
from src.training.trainer import (
    TrainingResult,
    fit_model,
    predict_logits,
    resolve_device
)
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed
from src.utils.timer import Timer


@dataclass
class DeepLearningExperimentResult:
    """
    Result of one original-scenario deep learning experiment.

    One experiment means one:
    - dataset
    - model
    - seed
    - split/fold
    combination.
    """

    dataset: str
    model_name: str
    scenario: str
    seed: int
    fold: Optional[int]
    split_name: str
    split_sizes: Dict[str, int]
    window_counts: Dict[str, int]
    preprocessing_summary: Dict[str, Any]
    training_result: TrainingResult
    evaluation_result: BinaryClassificationResult
    runtime_record: RuntimeRecord
    test_scores: np.ndarray
    test_labels: np.ndarray
    test_target_indices: np.ndarray

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible experiment summary.
        """
        return {
            "dataset": self.dataset,
            "model": self.model_name,
            "scenario": self.scenario,
            "seed": int(self.seed),
            "fold": self.fold,
            "split_name": self.split_name,
            "split_sizes": self.split_sizes,
            "window_counts": self.window_counts,
            "preprocessing": self.preprocessing_summary,
            "training": asdict(self.training_result),
            "runtime": self.runtime_record.as_dict(),
            "test_metrics": self.evaluation_result.as_dict()
        }


def prepare_deep_learning_partitions(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None
) -> tuple[Dict[str, SequenceWindowData], DeepLearningPreprocessor]:
    """
    Fit train-only normalization and create leakage-safe windows.

    The scaler is fitted exclusively on rows identified by train_indices.
    Window generation occurs after partitioning, using existing boundary
    logic for SKAB source_file groups and BATADAL time partitions.
    """
    preprocessor = DeepLearningPreprocessor()
    preprocessor.fit(X.iloc[split.train_indices])

    transformed_X = preprocessor.transform(X)

    window_config = config["windowing"]

    partitions = create_windows_for_split(
        X=transformed_X,
        y=y,
        split=split,
        sequence_length=int(window_config["sequence_length"]),
        groups=groups,
        timestamps=timestamps,
        label_strategy=window_config["label_strategy"]
    )

    return partitions, preprocessor


def save_deep_learning_result(
    result: DeepLearningExperimentResult,
    logger: ExperimentLogger
) -> None:
    """
    Save one result using the existing CSV + JSON logger.

    Only small numerical/reporting artifacts are saved here.
    Model checkpoints are managed separately by the training layer.
    """
    logger.save_params({
        "dataset": result.dataset,
        "model": result.model_name,
        "scenario": result.scenario,
        "seed": result.seed,
        "fold": result.fold,
        "split_name": result.split_name,
        "split_sizes": result.split_sizes,
        "window_counts": result.window_counts,
        "preprocessing": result.preprocessing_summary
    })

    logger.log_metrics(
        metrics=result.evaluation_result.as_dict(),
        step=result.fold,
        split="test"
    )

    logger.save_artifact_json(
        data={"history": result.training_result.history},
        filename="training_history"
    )

    logger.save_summary(result.summary())


def run_deep_learning_original_split(
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
    device: Optional[str | torch.device] = None,
    logger: Optional[ExperimentLogger] = None
) -> DeepLearningExperimentResult:
    """
    Run one deep learning model on one original-data split.

    This is intentionally one controlled run, not a full grid execution.
    Later orchestration can call this function for each model, fold and seed.
    """
    set_seed(seed)

    resolved_device = (
        torch.device(device)
        if device is not None
        else resolve_device(config)
    )

    windowed_partitions, preprocessor = prepare_deep_learning_partitions(
        X=X,
        y=y,
        split=split,
        config=config,
        groups=groups,
        timestamps=timestamps
    )

    training_config = config["training"]

    data_loaders = create_data_loaders(
        windowed_partitions=windowed_partitions,
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
        data_loaders=data_loaders,
        train_targets=windowed_partitions["train"].y,
        config=config,
        device=resolved_device
    )

    with Timer("test_inference") as inference_timer:
        test_logits, test_labels = predict_logits(
            model=model,
            data_loader=data_loaders["test"],
            device=resolved_device
        )

    test_probabilities = logits_to_probabilities(test_logits)

    threshold = float(
        config["evaluation"]["deep_learning_probability_threshold"]
    )
    zero_division = int(config["evaluation"]["zero_division"])

    evaluation_result = evaluate_binary_scores(
        y_true=test_labels.astype(int),
        scores=test_probabilities,
        threshold=threshold,
        score_name="probability",
        zero_division=zero_division
    )

    if inference_timer.elapsed is None:
        raise RuntimeError("Inference runtime measurement failed.")

    runtime_record = RuntimeRecord(
        dataset=dataset_name,
        model=model_name,
        scenario="original",
        seed=seed,
        fold=fold,
        training_seconds=training_result.training_time_seconds,
        inference_seconds=float(inference_timer.elapsed),
        sample_count=int(len(test_labels))
    )

    result = DeepLearningExperimentResult(
        dataset=dataset_name,
        model_name=model_name,
        scenario="original",
        seed=seed,
        fold=fold,
        split_name=split.split_name,
        split_sizes=split.sizes(),
        window_counts={
            name: len(window_data)
            for name, window_data in windowed_partitions.items()
        },
        preprocessing_summary=preprocessor.summary(),
        training_result=training_result,
        evaluation_result=evaluation_result,
        runtime_record=runtime_record,
        test_scores=test_probabilities,
        test_labels=test_labels.astype(int),
        test_target_indices=windowed_partitions["test"].target_indices
    )

    if logger is not None:
        save_deep_learning_result(result=result, logger=logger)

    return result


def run_batadal_original_deep_learning(
    df: pd.DataFrame,
    config: Dict[str, Any],
    model_name: str,
    seed: int,
    device: Optional[str | torch.device] = None,
    logger: Optional[ExperimentLogger] = None
) -> DeepLearningExperimentResult:
    """
    Run one original-scenario BATADAL deep learning experiment.

    BATADAL uses one chronological 60/20/20 split.
    """
    X, y, timestamps = prepare_batadal_features_target(df, config)
    split = create_batadal_time_split(df, config)

    return run_deep_learning_original_split(
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
        device=device,
        logger=logger
    )


def run_skab_original_deep_learning_fold(
    df: pd.DataFrame,
    config: Dict[str, Any],
    model_name: str,
    seed: int,
    fold: int,
    device: Optional[str | torch.device] = None,
    logger: Optional[ExperimentLogger] = None
) -> DeepLearningExperimentResult:
    """
    Run one original-scenario SKAB deep learning fold.

    The function intentionally runs one selected fold. Full five-fold,
    five-seed execution will later call this controlled unit repeatedly.
    """
    X, y, groups = prepare_skab_features_target(df, config)
    splits = create_skab_nested_splits(df, config)

    if fold < 1 or fold > len(splits):
        raise ValueError(
            f"fold must be between 1 and {len(splits)}, found {fold}."
        )

    split = splits[fold - 1]

    return run_deep_learning_original_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="SKAB",
        model_name=model_name,
        seed=seed,
        fold=fold,
        groups=groups,
        timestamps=None,
        device=device,
        logger=logger
    )