"""
Original-scenario experiment orchestration for probabilistic automata.

Pipeline
--------
features -> train-only scaler/PCA -> PC1 sequences
-> sliding context windows -> PAA words -> train-fitted SAX states
-> normal-train probabilistic automata
-> validation threshold calibration
-> test scoring and explainability
-> metrics and runtime/loggable result

Important leakage rules
-----------------------
- Scaler and PCA are fitted only on training features.
- SAX normalization statistics are fitted only on training PAA values.
- Automata transitions are learned only from normal training runs.
- Threshold is calibrated only on validation decisions.
- Test labels are used only for final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.automata.calibration import (
    ThresholdCalibrationResult,
    calibrate_f1_threshold
)
from src.automata.explainability import (
    AutomataDecisionExplanation,
    explain_automata_decision
)
from src.automata.levenshtein import LevenshteinStateMapper
from src.automata.paa import paa_transform
from src.automata.probabilistic_automata import ProbabilisticAutomata
from src.automata.sax import SAXDiscretizer
from src.data.load_batadal import prepare_batadal_features_target
from src.data.load_skab import prepare_skab_features_target
from src.data.preprocessing import (
    AutomataPreprocessor,
    build_automata_preprocessor_from_config
)
from src.data.splitting import (
    DatasetSplit,
    create_batadal_time_split,
    create_skab_nested_splits
)
from src.evaluation.metrics import (
    BinaryClassificationResult,
    evaluate_binary_scores
)
from src.evaluation.runtime import RuntimeRecord
from src.utils.logger import ExperimentLogger
from src.utils.timer import Timer


@dataclass(frozen=True)
class PC1Sequence:
    """
    One partition-safe PC1 time series.

    SKAB produces one sequence per source_file inside each partition.
    BATADAL produces one chronological sequence per partition.
    """

    sequence_id: str
    values: np.ndarray
    labels: np.ndarray
    target_indices: np.ndarray
    timestamps: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PAAWordSequence:
    """
    PAA representations created from rolling PC1 context windows.
    """

    sequence_id: str
    paa_values: np.ndarray
    labels: np.ndarray
    target_indices: np.ndarray
    timestamps: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SymbolicStateSequence:
    """
    SAX state sequence aligned with final-time-step labels.
    """

    sequence_id: str
    states: tuple[str, ...]
    labels: np.ndarray
    target_indices: np.ndarray
    timestamps: Optional[np.ndarray] = None


@dataclass(frozen=True)
class AutomataDecisionSet:
    """
    Per-transition automata decision data.
    """

    scores: np.ndarray
    labels: np.ndarray
    target_indices: np.ndarray
    sequence_ids: np.ndarray
    explanations: tuple[AutomataDecisionExplanation, ...] = ()


@dataclass
class AutomataExperimentResult:
    """
    Result of one original-scenario automata experiment.
    """

    dataset: str
    scenario: str
    fold: Optional[int]
    split_name: str
    split_sizes: Dict[str, int]
    state_counts: Dict[str, int]
    preprocessing_summary: Dict[str, Any]
    sax_summary: Dict[str, Any]
    automata_summary: Dict[str, Any]
    calibration_result: ThresholdCalibrationResult
    evaluation_result: BinaryClassificationResult
    runtime_record: RuntimeRecord
    test_scores: np.ndarray
    test_labels: np.ndarray
    test_target_indices: np.ndarray
    test_explanations: tuple[AutomataDecisionExplanation, ...]
    automata: ProbabilisticAutomata

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible experiment summary.
        """
        return {
            "dataset": self.dataset,
            "model": "automata",
            "scenario": self.scenario,
            "fold": self.fold,
            "split_name": self.split_name,
            "split_sizes": self.split_sizes,
            "state_counts": self.state_counts,
            "preprocessing": self.preprocessing_summary,
            "sax": self.sax_summary,
            "automata": self.automata_summary,
            "calibration": self.calibration_result.summary(),
            "runtime": self.runtime_record.as_dict(),
            "test_metrics": self.evaluation_result.as_dict()
        }


def prepare_pc1_partition_sequences(
    pc1: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None
) -> Dict[str, Dict[str, PC1Sequence]]:
    """
    Convert split rows into partition-safe PC1 sequences.

    Windows are never allowed to cross:
    - train/validation/test boundaries
    - SKAB source_file boundaries when groups are supplied
    """
    if list(pc1.columns) != ["PC1"]:
        raise ValueError("pc1 dataframe must contain exactly one 'PC1' column.")

    partition_indices = {
        "train": split.train_indices,
        "validation": split.validation_indices,
        "test": split.test_indices
    }

    prepared: Dict[str, Dict[str, PC1Sequence]] = {}

    for partition_name, raw_indices in partition_indices.items():
        indices = np.asarray(raw_indices, dtype=int)
        partition_sequences: Dict[str, PC1Sequence] = {}

        if groups is None:
            sequence_groups = [(partition_name, indices)]
        else:
            group_values = groups.iloc[indices].to_numpy()
            sequence_groups = []

            for group_value in pd.unique(group_values):
                group_indices = indices[group_values == group_value]
                sequence_groups.append((str(group_value), group_indices))

        for sequence_name, sequence_indices in sequence_groups:
            ordered_indices = np.sort(np.asarray(sequence_indices, dtype=int))

            sequence_id = f"{partition_name}::{sequence_name}"

            sequence_timestamps = (
                timestamps.iloc[ordered_indices].to_numpy()
                if timestamps is not None
                else None
            )

            partition_sequences[sequence_id] = PC1Sequence(
                sequence_id=sequence_id,
                values=pc1.iloc[ordered_indices]["PC1"].to_numpy(
                    dtype=np.float64
                ),
                labels=y.iloc[ordered_indices].to_numpy(dtype=int),
                target_indices=pc1.index.to_numpy()[ordered_indices],
                timestamps=sequence_timestamps
            )

        prepared[partition_name] = partition_sequences

    return prepared


def create_paa_word_sequence(
    sequence: PC1Sequence,
    context_length: int,
    word_size: int
) -> PAAWordSequence:
    """
    Create rolling PAA words from one independent PC1 sequence.

    Each word uses the most recent context_length observations and its
    label is the final time-step label of that context.
    """
    if context_length <= 0:
        raise ValueError("context_length must be greater than zero.")

    if word_size <= 0:
        raise ValueError("word_size must be greater than zero.")

    if word_size > context_length:
        raise ValueError("word_size cannot exceed context_length.")

    if len(sequence.values) < context_length:
        raise ValueError(
            f"Sequence '{sequence.sequence_id}' is shorter than context_length."
        )

    paa_values = []
    labels = []
    target_indices = []
    output_timestamps = []

    for end_position in range(context_length - 1, len(sequence.values)):
        start_position = end_position - context_length + 1

        context_values = sequence.values[start_position:end_position + 1]

        paa_values.append(
            paa_transform(context_values, n_segments=word_size)
        )
        labels.append(int(sequence.labels[end_position]))
        target_indices.append(sequence.target_indices[end_position])

        if sequence.timestamps is not None:
            output_timestamps.append(sequence.timestamps[end_position])

    return PAAWordSequence(
        sequence_id=sequence.sequence_id,
        paa_values=np.asarray(paa_values, dtype=np.float64),
        labels=np.asarray(labels, dtype=int),
        target_indices=np.asarray(target_indices),
        timestamps=(
            np.asarray(output_timestamps)
            if sequence.timestamps is not None
            else None
        )
    )


def create_paa_words_for_partitions(
    sequences: Dict[str, Dict[str, PC1Sequence]],
    context_length: int,
    word_size: int
) -> Dict[str, Dict[str, PAAWordSequence]]:
    """
    Create PAA word sequences for every data partition.
    """
    output: Dict[str, Dict[str, PAAWordSequence]] = {}

    for partition_name, partition_sequences in sequences.items():
        output[partition_name] = {}

        for sequence_id, sequence in partition_sequences.items():
            if len(sequence.values) < context_length:
                continue

            output[partition_name][sequence_id] = create_paa_word_sequence(
                sequence=sequence,
                context_length=context_length,
                word_size=word_size
            )

        if not output[partition_name]:
            raise ValueError(
                f"No PAA words could be created for partition "
                f"'{partition_name}'."
            )

    return output


def fit_sax_from_training_words(
    training_words: Dict[str, PAAWordSequence],
    alphabet_size: int
) -> SAXDiscretizer:
    """
    Fit SAX normalization using only training PAA values.
    """
    flattened_training_values = np.concatenate(
        [
            word_sequence.paa_values.reshape(-1)
            for word_sequence in training_words.values()
        ]
    )

    discretizer = SAXDiscretizer(alphabet_size=alphabet_size)
    discretizer.fit(flattened_training_values)

    return discretizer


def encode_symbolic_states(
    word_sequences: Dict[str, PAAWordSequence],
    discretizer: SAXDiscretizer
) -> Dict[str, SymbolicStateSequence]:
    """
    Convert PAA word sequences into SAX state strings.
    """
    encoded: Dict[str, SymbolicStateSequence] = {}

    for sequence_id, word_sequence in word_sequences.items():
        state_symbols = discretizer.transform(
            word_sequence.paa_values.reshape(-1)
        ).reshape(word_sequence.paa_values.shape)

        states = tuple(
            "".join(symbol_row.tolist())
            for symbol_row in state_symbols
        )

        encoded[sequence_id] = SymbolicStateSequence(
            sequence_id=sequence_id,
            states=states,
            labels=word_sequence.labels.copy(),
            target_indices=word_sequence.target_indices.copy(),
            timestamps=(
                word_sequence.timestamps.copy()
                if word_sequence.timestamps is not None
                else None
            )
        )

    return encoded


def score_transition_decisions(
    automata: ProbabilisticAutomata,
    mapper: LevenshteinStateMapper,
    sequences: Dict[str, SymbolicStateSequence],
    threshold: Optional[float] = None
) -> AutomataDecisionSet:
    """
    Score each transition ending at a current symbolic state.

    The target label for transition previous_state -> current_state
    is the label associated with current_state.
    """
    scores = []
    labels = []
    target_indices = []
    sequence_ids = []
    explanations = []

    for sequence_id, sequence in sequences.items():
        if len(sequence.states) < 2:
            continue

        for state_position in range(1, len(sequence.states)):
            observed_pair = [
                sequence.states[state_position - 1],
                sequence.states[state_position]
            ]

            if threshold is None:
                mapped_states, _ = mapper.map_sequence(observed_pair)

                path_score = automata.score_path(
                    states=mapped_states,
                    sequence_id=sequence_id
                )

                score = path_score.mean_negative_log_probability
            else:
                explanation = explain_automata_decision(
                    automata=automata,
                    mapper=mapper,
                    observed_states=observed_pair,
                    threshold=threshold,
                    sequence_id=sequence_id
                )

                explanations.append(explanation)
                score = explanation.anomaly_score

            scores.append(float(score))
            labels.append(int(sequence.labels[state_position]))
            target_indices.append(sequence.target_indices[state_position])
            sequence_ids.append(sequence_id)

    if not scores:
        raise ValueError("No automata transition decisions could be scored.")

    return AutomataDecisionSet(
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=int),
        target_indices=np.asarray(target_indices),
        sequence_ids=np.asarray(sequence_ids, dtype=object),
        explanations=tuple(explanations)
    )


def save_automata_result(
    result: AutomataExperimentResult,
    logger: ExperimentLogger
) -> None:
    """
    Save summary, metrics, calibration and explanations.
    """
    logger.save_params({
        "dataset": result.dataset,
        "model": "automata",
        "scenario": result.scenario,
        "fold": result.fold,
        "split_name": result.split_name,
        "split_sizes": result.split_sizes,
        "state_counts": result.state_counts,
        "preprocessing": result.preprocessing_summary,
        "sax": result.sax_summary,
        "automata": result.automata_summary
    })

    logger.log_metrics(
        metrics=result.evaluation_result.as_dict(),
        step=result.fold,
        split="test"
    )

    logger.save_artifact_json(
        data=result.calibration_result.summary(),
        filename="threshold_calibration"
    )

    logger.save_artifact_json(
        data={
            "explanations": [
                explanation.as_dict()
                for explanation in result.test_explanations
            ]
        },
        filename="test_explanations"
    )

    logger.save_summary(result.summary())


def run_automata_original_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    dataset_name: str,
    fold: Optional[int] = None,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None,
    logger: Optional[ExperimentLogger] = None
) -> AutomataExperimentResult:
    """
    Run one original-scenario probabilistic automata experiment.
    """
    automata_config = config["automata"]

    context_length = int(automata_config["context_length"])
    word_size = int(automata_config["default_window_size"])
    alphabet_size = int(automata_config["default_alphabet_size"])
    smoothing = float(automata_config["smoothing"])

    with Timer("automata_fit_and_calibration") as training_timer:
        preprocessor: AutomataPreprocessor = (
            build_automata_preprocessor_from_config(config)
        )

        preprocessor.fit(X.iloc[split.train_indices])
        pc1 = preprocessor.transform(X)

        partition_sequences = prepare_pc1_partition_sequences(
            pc1=pc1,
            y=y,
            split=split,
            groups=groups,
            timestamps=timestamps
        )

        paa_partitions = create_paa_words_for_partitions(
            sequences=partition_sequences,
            context_length=context_length,
            word_size=word_size
        )

        discretizer = fit_sax_from_training_words(
            training_words=paa_partitions["train"],
            alphabet_size=alphabet_size
        )

        symbolic_partitions = {
            partition_name: encode_symbolic_states(
                word_sequences=partition_words,
                discretizer=discretizer
            )
            for partition_name, partition_words in paa_partitions.items()
        }

        train_state_sequences = {
            sequence_id: sequence.states
            for sequence_id, sequence in symbolic_partitions["train"].items()
        }

        train_label_sequences = {
            sequence_id: sequence.labels
            for sequence_id, sequence in symbolic_partitions["train"].items()
        }

        automata = ProbabilisticAutomata(smoothing=smoothing)
        automata.fit_from_labeled_sequences(
            pattern_sequences=train_state_sequences,
            label_sequences=train_label_sequences
        )

        mapper = LevenshteinStateMapper(automata.states_)

        validation_decisions = score_transition_decisions(
            automata=automata,
            mapper=mapper,
            sequences=symbolic_partitions["validation"],
            threshold=None
        )

        calibration_result = calibrate_f1_threshold(
            validation_scores=validation_decisions.scores,
            validation_labels=validation_decisions.labels
        )

    if training_timer.elapsed is None:
        raise RuntimeError("Automata training runtime measurement failed.")

    with Timer("automata_test_inference") as inference_timer:
        test_decisions = score_transition_decisions(
            automata=automata,
            mapper=mapper,
            sequences=symbolic_partitions["test"],
            threshold=calibration_result.threshold
        )

        evaluation_result = evaluate_binary_scores(
            y_true=test_decisions.labels,
            scores=test_decisions.scores,
            threshold=calibration_result.threshold,
            score_name="automata_anomaly_score",
            zero_division=int(config["evaluation"]["zero_division"])
        )

    if inference_timer.elapsed is None:
        raise RuntimeError("Automata inference runtime measurement failed.")

    runtime_record = RuntimeRecord(
        dataset=dataset_name,
        model="automata",
        scenario="original",
        seed=None,
        fold=fold,
        training_seconds=float(training_timer.elapsed),
        inference_seconds=float(inference_timer.elapsed),
        sample_count=int(len(test_decisions.labels))
    )

    result = AutomataExperimentResult(
        dataset=dataset_name,
        scenario="original",
        fold=fold,
        split_name=split.split_name,
        split_sizes=split.sizes(),
        state_counts={
            partition_name: int(
                sum(len(sequence.states) for sequence in sequences.values())
            )
            for partition_name, sequences in symbolic_partitions.items()
        },
        preprocessing_summary=preprocessor.summary(),
        sax_summary=discretizer.summary(),
        automata_summary=automata.summary(),
        calibration_result=calibration_result,
        evaluation_result=evaluation_result,
        runtime_record=runtime_record,
        test_scores=test_decisions.scores,
        test_labels=test_decisions.labels,
        test_target_indices=test_decisions.target_indices,
        test_explanations=test_decisions.explanations,
        automata=automata
    )

    if logger is not None:
        save_automata_result(result=result, logger=logger)

    return result


def run_batadal_original_automata(
    df: pd.DataFrame,
    config: Dict[str, Any],
    logger: Optional[ExperimentLogger] = None
) -> AutomataExperimentResult:
    """
    Run one original BATADAL automata experiment.
    """
    X, y, timestamps = prepare_batadal_features_target(df, config)
    split = create_batadal_time_split(df, config)

    return run_automata_original_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="BATADAL",
        fold=None,
        groups=None,
        timestamps=timestamps,
        logger=logger
    )


def run_skab_original_automata_fold(
    df: pd.DataFrame,
    config: Dict[str, Any],
    fold: int,
    logger: Optional[ExperimentLogger] = None
) -> AutomataExperimentResult:
    """
    Run one selected SKAB automata fold.
    """
    X, y, groups = prepare_skab_features_target(df, config)
    splits = create_skab_nested_splits(df, config)

    if fold < 1 or fold > len(splits):
        raise ValueError(
            f"fold must be between 1 and {len(splits)}, found {fold}."
        )

    return run_automata_original_split(
        X=X,
        y=y,
        split=splits[fold - 1],
        config=config,
        dataset_name="SKAB",
        fold=fold,
        groups=groups,
        timestamps=None,
        logger=logger
    )