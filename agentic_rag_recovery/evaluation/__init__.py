"""Evaluation subsystem: datasets, metrics and the experiment engine."""

from .datasets import (
    PUBLIC_DATASETS,
    DemoDataset,
    DemoQuestion,
    list_public_datasets,
    load_demo_dataset,
    load_public_dataset,
)
from .metrics import (
    RunMetrics,
    aggregate,
    class_distribution,
    compare_modes,
    diagnosis_correct,
    evaluate_run,
    severity_distribution,
    stage_distribution,
)
from .runner import ExperimentResult, ExperimentRunner, run_experiment_by_name

__all__ = [
    "DemoDataset",
    "DemoQuestion",
    "load_demo_dataset",
    "load_public_dataset",
    "list_public_datasets",
    "PUBLIC_DATASETS",
    "RunMetrics",
    "evaluate_run",
    "aggregate",
    "compare_modes",
    "diagnosis_correct",
    "severity_distribution",
    "stage_distribution",
    "class_distribution",
    "ExperimentRunner",
    "ExperimentResult",
    "run_experiment_by_name",
]