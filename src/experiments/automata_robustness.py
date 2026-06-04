"""
Robustness and parameter-analysis orchestration for probabilistic automata.

Correct Gaussian-noise protocol
-------------------------------
1. Fit scaler and PCA only on clean training features.
2. Fit SAX normalization only on clean training PAA values.
3. Learn automata transitions only from clean normal training runs.
4. Calibrate anomaly threshold only on clean validation decisions.
5. Evaluate the same fitted pipeline on original and noisy test data.

No component is re-fitted for individual Gaussian-noise levels.
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
from src.automata.levenshtein import LevenshteinStateMapper
from src.automata.probabilistic_automata import ProbabilisticAutomata
from src.automata.run_experiment import (
    AutomataDecisionSet,
    AutomataExperimentResult,
    SymbolicStateSequence,
    create_paa_words_for_partitions,
    encode_symbolic_states,
    fit_sax_from_training_words,
    prepare_pc1_partition_sequences,
    run_automata_original_split,
    score_transition_decisions
)
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
from src.experiments.scenarios import (
    AutomataParameterSetting,
    GaussianNoiseSetting,
    UnseenPatternAnalysisSummary,
    build_config_for_automata_setting,
    build_gaussian_noise_settings,
    create_automata_parameter_grid,
    inject_gaussian_noise_into_test_rows,
    summarize_unseen_automata_decisions
)
from src.utils.timer import Timer

from src.automata.explainability import AutomataDecisionExplanation


@dataclass
class FittedAutomataPipeline:
    """
    Clean-data fitted automata components reused across test scenarios.
    """

    preprocessor: AutomataPreprocessor
    discretizer: SAXDiscretizer
    automata: ProbabilisticAutomata
    mapper: LevenshteinStateMapper
    calibration_result: ThresholdCalibrationResult
    clean_symbolic_partitions: Dict[str, Dict[str, SymbolicStateSequence]]
    context_length: int
    word_size: int
    alphabet_size: int
    training_seconds: float

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible fitted pipeline summary.
        """
        return {
            "context_length": self.context_length,
            "window_size": self.word_size,
            "alphabet_size": self.alphabet_size,
            "preprocessing": self.preprocessor.summary(),
            "sax": self.discretizer.summary(),
            "automata": self.automata.summary(),
            "calibration": self.calibration_result.summary(),
            "training_seconds": self.training_seconds
        }


@dataclass(frozen=True)
class AutomataScenarioEvaluation:
    """
    Result for one original or Gaussian-noise automata test scenario.
    """

    scenario: str
    noise_level: Optional[float]
    clean_pipeline_reused: bool
    evaluation_result: BinaryClassificationResult
    runtime_record: RuntimeRecord
    unseen_summary: UnseenPatternAnalysisSummary
    test_scores: np.ndarray
    test_labels: np.ndarray
    test_target_indices: np.ndarray
    test_explanations: tuple[AutomataDecisionExplanation, ...]

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible scenario summary.
        """
        return {
            "scenario": self.scenario,
            "noise_level": self.noise_level,
            "clean_pipeline_reused": self.clean_pipeline_reused,
            "runtime": self.runtime_record.as_dict(),
            "metrics": self.evaluation_result.as_dict(),
            "unseen_analysis": self.unseen_summary.as_dict(),
            "explanation_count": len(self.test_explanations)
        }


@dataclass
class AutomataRobustnessResult:
    """
    Original and Gaussian-noise evaluation of one fitted automata pipeline.
    """

    dataset: str
    fold: Optional[int]
    split_name: str
    split_sizes: Dict[str, int]
    fitted_pipeline: FittedAutomataPipeline
    scenario_results: Dict[str, AutomataScenarioEvaluation]

    def summary(self) -> Dict[str, Any]:
        """
        Return JSON-compatible robustness summary.
        """
        return {
            "dataset": self.dataset,
            "model": "automata",
            "fold": self.fold,
            "split_name": self.split_name,
            "split_sizes": self.split_sizes,
            "protocol": {
                "fitted_on": "clean_training_partition",
                "threshold_calibrated_on": "clean_validation_partition",
                "noise_applied_to": "test_only",
                "refit_for_noise_levels": False
            },
            "fitted_pipeline": self.fitted_pipeline.summary(),
            "scenario_results": {
                scenario_name: scenario_result.summary()
                for scenario_name, scenario_result
                in self.scenario_results.items()
            }
        }


@dataclass
class AutomataParameterSweepResult:
    """
    Stores clean original-data automata parameter analysis results.
    """

    dataset: str
    fold: Optional[int]
    results: Dict[str, AutomataExperimentResult]
    settings: Dict[str, AutomataParameterSetting]

    def results_table(self) -> pd.DataFrame:
        """
        Return one row per automata parameter combination.
        """
        rows = []

        for scenario_name, result in self.results.items():
            setting = self.settings[scenario_name]
            metrics = result.evaluation_result.as_dict()

            rows.append({
                "dataset": self.dataset,
                "fold": self.fold,
                "scenario": scenario_name,
                "context_length": setting.context_length,
                "window_size": setting.window_size,
                "alphabet_size": setting.alphabet_size,
                "state_count": result.automata_summary["state_count"],
                "threshold": result.calibration_result.threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "training_seconds": result.runtime_record.training_seconds,
                "inference_seconds": result.runtime_record.inference_seconds
            })

        return pd.DataFrame(rows)


def _create_symbolic_partitions_with_fitted_components(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    preprocessor: AutomataPreprocessor,
    discretizer: SAXDiscretizer,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None
) -> Dict[str, Dict[str, SymbolicStateSequence]]:
    """
    Convert raw features into symbolic states using already fitted components.

    This function performs transform operations only:
    - no scaler fitting
    - no PCA fitting
    - no SAX fitting
    """
    automata_config = config["automata"]

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
        context_length=int(automata_config["context_length"]),
        word_size=int(automata_config["default_window_size"])
    )

    return {
        partition_name: encode_symbolic_states(
            word_sequences=partition_words,
            discretizer=discretizer
        )
        for partition_name, partition_words in paa_partitions.items()
    }


def fit_clean_automata_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None
) -> FittedAutomataPipeline:
    """
    Fit automata components using clean training and validation data only.

    Training usage
    --------------
    - Scaler/PCA: clean train features only.
    - SAX normalization: clean train PAA values only.
    - Automata transitions: normal runs within clean train only.
    - Threshold calibration: clean validation decisions only.
    """
    automata_config = config["automata"]

    context_length = int(automata_config["context_length"])
    word_size = int(automata_config["default_window_size"])
    alphabet_size = int(automata_config["default_alphabet_size"])
    smoothing = float(automata_config["smoothing"])

    with Timer("clean_automata_fit_and_calibration") as training_timer:
        preprocessor = build_automata_preprocessor_from_config(config)
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

        train_states = {
            sequence_id: sequence.states
            for sequence_id, sequence in symbolic_partitions["train"].items()
        }

        train_labels = {
            sequence_id: sequence.labels
            for sequence_id, sequence in symbolic_partitions["train"].items()
        }

        automata = ProbabilisticAutomata(smoothing=smoothing)
        automata.fit_from_labeled_sequences(
            pattern_sequences=train_states,
            label_sequences=train_labels
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
        raise RuntimeError("Automata clean fitting runtime measurement failed.")

    return FittedAutomataPipeline(
        preprocessor=preprocessor,
        discretizer=discretizer,
        automata=automata,
        mapper=mapper,
        calibration_result=calibration_result,
        clean_symbolic_partitions=symbolic_partitions,
        context_length=context_length,
        word_size=word_size,
        alphabet_size=alphabet_size,
        training_seconds=float(training_timer.elapsed)
    )


def _evaluate_fitted_automata_on_test_partition(
    fitted_pipeline: FittedAutomataPipeline,
    symbolic_test_sequences: Dict[str, SymbolicStateSequence],
    config: Dict[str, Any],
    dataset_name: str,
    fold: Optional[int],
    scenario: str,
    noise_level: Optional[float],
    training_seconds: float,
    clean_pipeline_reused: bool
) -> AutomataScenarioEvaluation:
    """
    Evaluate a fitted automata pipeline on one test scenario.
    """
    threshold = fitted_pipeline.calibration_result.threshold

    with Timer(f"{scenario}_automata_test_inference") as inference_timer:
        decisions: AutomataDecisionSet = score_transition_decisions(
            automata=fitted_pipeline.automata,
            mapper=fitted_pipeline.mapper,
            sequences=symbolic_test_sequences,
            threshold=threshold
        )

        evaluation_result = evaluate_binary_scores(
            y_true=decisions.labels,
            scores=decisions.scores,
            threshold=threshold,
            score_name="automata_anomaly_score",
            zero_division=int(config["evaluation"]["zero_division"])
        )

        unseen_summary = summarize_unseen_automata_decisions(
            explanations=decisions.explanations,
            labels=decisions.labels,
            scores=decisions.scores,
            threshold=threshold,
            zero_division=int(config["evaluation"]["zero_division"])
        )

    if inference_timer.elapsed is None:
        raise RuntimeError("Automata scenario inference timing failed.")

    runtime_record = RuntimeRecord(
        dataset=dataset_name,
        model="automata",
        scenario=scenario,
        seed=None,
        fold=fold,
        training_seconds=float(training_seconds),
        inference_seconds=float(inference_timer.elapsed),
        sample_count=int(len(decisions.labels))
    )

    return AutomataScenarioEvaluation(
        scenario=scenario,
        noise_level=noise_level,
        clean_pipeline_reused=clean_pipeline_reused,
        evaluation_result=evaluation_result,
        runtime_record=runtime_record,
        unseen_summary=unseen_summary,
        test_scores=decisions.scores,
        test_labels=decisions.labels,
        test_target_indices=decisions.target_indices,
        test_explanations=decisions.explanations
    )


def run_automata_gaussian_robustness_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    dataset_name: str,
    fold: Optional[int] = None,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None
) -> AutomataRobustnessResult:
    """
    Fit one clean automata pipeline and evaluate original/noisy test scenarios.
    """
    fitted_pipeline = fit_clean_automata_pipeline(
        X=X,
        y=y,
        split=split,
        config=config,
        groups=groups,
        timestamps=timestamps
    )

    scenario_results: Dict[str, AutomataScenarioEvaluation] = {}

    scenario_results["original"] = _evaluate_fitted_automata_on_test_partition(
        fitted_pipeline=fitted_pipeline,
        symbolic_test_sequences=(
            fitted_pipeline.clean_symbolic_partitions["test"]
        ),
        config=config,
        dataset_name=dataset_name,
        fold=fold,
        scenario="original",
        noise_level=None,
        training_seconds=fitted_pipeline.training_seconds,
        clean_pipeline_reused=False
    )

    original_result = scenario_results["original"]

    for setting in build_gaussian_noise_settings(config):
        noisy_X = inject_gaussian_noise_into_test_rows(
            X=X,
            split=split,
            setting=setting
        )

        noisy_symbolic_partitions = (
            _create_symbolic_partitions_with_fitted_components(
                X=noisy_X,
                y=y,
                split=split,
                config=config,
                preprocessor=fitted_pipeline.preprocessor,
                discretizer=fitted_pipeline.discretizer,
                groups=groups,
                timestamps=timestamps
            )
        )

        noisy_result = _evaluate_fitted_automata_on_test_partition(
            fitted_pipeline=fitted_pipeline,
            symbolic_test_sequences=noisy_symbolic_partitions["test"],
            config=config,
            dataset_name=dataset_name,
            fold=fold,
            scenario=setting.scenario_name,
            noise_level=setting.level,
            training_seconds=0.0,
            clean_pipeline_reused=True
        )

        if not np.array_equal(
            original_result.test_target_indices,
            noisy_result.test_target_indices
        ):
            raise RuntimeError(
                "Noise scenario changed automata test target-index alignment."
            )

        if not np.array_equal(
            original_result.test_labels,
            noisy_result.test_labels
        ):
            raise RuntimeError(
                "Noise scenario changed automata test labels unexpectedly."
            )

        scenario_results[setting.scenario_name] = noisy_result

    return AutomataRobustnessResult(
        dataset=dataset_name,
        fold=fold,
        split_name=split.split_name,
        split_sizes=split.sizes(),
        fitted_pipeline=fitted_pipeline,
        scenario_results=scenario_results
    )


def run_automata_parameter_analysis_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: DatasetSplit,
    config: Dict[str, Any],
    dataset_name: str,
    fold: Optional[int] = None,
    groups: Optional[pd.Series] = None,
    timestamps: Optional[pd.Series] = None
) -> AutomataParameterSweepResult:
    """
    Evaluate clean original-data automata behavior across parameter settings.

    Each parameter combination is independently fitted and calibrated using:
    - clean train data for preprocessing/SAX/automata fitting
    - clean validation data for threshold calibration
    - clean test data for final metric evaluation
    """
    settings = create_automata_parameter_grid(config)

    results: Dict[str, AutomataExperimentResult] = {}
    settings_by_name: Dict[str, AutomataParameterSetting] = {}

    for setting in settings:
        adjusted_config = build_config_for_automata_setting(
            config=config,
            setting=setting
        )

        scenario_name = setting.scenario_name

        results[scenario_name] = run_automata_original_split(
            X=X,
            y=y,
            split=split,
            config=adjusted_config,
            dataset_name=dataset_name,
            fold=fold,
            groups=groups,
            timestamps=timestamps,
            logger=None
        )

        settings_by_name[scenario_name] = setting

    return AutomataParameterSweepResult(
        dataset=dataset_name,
        fold=fold,
        results=results,
        settings=settings_by_name
    )


def run_batadal_automata_gaussian_robustness(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> AutomataRobustnessResult:
    """
    Run BATADAL automata Gaussian-noise robustness evaluation.
    """
    X, y, timestamps = prepare_batadal_features_target(df, config)
    split = create_batadal_time_split(df, config)

    return run_automata_gaussian_robustness_split(
        X=X,
        y=y,
        split=split,
        config=config,
        dataset_name="BATADAL",
        fold=None,
        groups=None,
        timestamps=timestamps
    )


def run_skab_automata_gaussian_robustness_fold(
    df: pd.DataFrame,
    config: Dict[str, Any],
    fold: int
) -> AutomataRobustnessResult:
    """
    Run one SKAB automata fold under Gaussian-noise robustness evaluation.
    """
    X, y, groups = prepare_skab_features_target(df, config)
    splits = create_skab_nested_splits(df, config)

    if fold < 1 or fold > len(splits):
        raise ValueError(
            f"fold must be between 1 and {len(splits)}, found {fold}."
        )

    return run_automata_gaussian_robustness_split(
        X=X,
        y=y,
        split=splits[fold - 1],
        config=config,
        dataset_name="SKAB",
        fold=fold,
        groups=groups,
        timestamps=None
    )