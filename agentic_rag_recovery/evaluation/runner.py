"""Experiment engine.

Runs reproducible experiments:

    BASELINE (mode 1)  →  AGENTIC (mode 2)  →  FAULT-INJECTED + RECOVERY (mode 3)
    →  DIAGNOSE → RECOVER → VERIFY → EVALUATE

Every artefact (run, diagnosis, recovery attempt, fault, metric row) is written
to SQLite so any figure in the report can be traced back to the request that
produced it.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from ..bootstrap import Application
from ..evaluation.datasets import DemoQuestion, DemoDataset, load_demo_dataset
from ..evaluation.metrics import (
    ExperimentMetrics,
    RunMetrics,
    aggregate,
    compare_modes,
    evaluate_run,
)
from ..models import ExperimentMetrics as ExperimentMetricsModel, PipelineRun, RunMode, new_id
from ..observability.trace import get_logger
from ..settings import Settings, get_settings
from ..faults.definitions import FaultDefinition

logger = get_logger(__name__)


DEFAULT_EXPERIMENT_CONFIG: Dict[str, Any] = {
    "name": "unnamed",
    "description": "",
    "seed": 42,
    "repetitions": 1,
    "modes": ["mode1_normal_rag", "mode2_agentic_no_recovery", "mode3_agentic_with_recovery"],
    "fault_type": None,
    "fault_parameters": {},
    "question_ids": None,
    "limit": None,
    "retrieval": {},
    "recovery": {},
    "model": None,
    "dataset": "demo",
}


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    config: Dict[str, Any]
    mode1: Optional[ExperimentMetrics] = None
    mode2: Optional[ExperimentMetrics] = None
    mode3: Optional[ExperimentMetrics] = None
    comparison: Dict[str, Any] = field(default_factory=dict)
    label: str = "DEMO / MOCK RESULTS"
    duration_s: float = 0.0
    run_ids: Dict[str, List[str]] = field(default_factory=lambda: {"mode1": [], "mode2": [], "mode3": []})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "config": self.config,
            "label": self.label,
            "duration_s": round(self.duration_s, 3),
            "mode1_normal_rag": self.mode1.model_dump() if self.mode1 else None,
            "mode2_agentic_no_recovery": self.mode2.model_dump() if self.mode2 else None,
            "mode3_agentic_with_recovery": self.mode3.model_dump() if self.mode3 else None,
            "comparison": self.comparison,
            "run_ids": self.run_ids,
        }


class ExperimentRunner:
    """Executes one experiment configuration against the live application."""

    def __init__(self, app: Application, settings: Optional[Settings] = None):
        self.app = app
        self.settings = settings or app.settings

    # ------------------------------------------------------------------
    def load_config(self, name_or_path: str) -> Dict[str, Any]:
        """Load an experiment config from ``experiments/configs`` or a path."""
        candidate = Path(name_or_path)
        if not candidate.exists():
            candidate = Path(self.settings.paths.experiment_config_dir) / f"{name_or_path}.yaml"
        if not candidate.exists():
            for suffix in (".yaml", ".yml"):
                candidate = Path(self.settings.paths.experiment_config_dir) / f"{name_or_path}{suffix}"
                if candidate.exists():
                    break
        if not candidate.exists():
            raise FileNotFoundError(f"experiment config not found: {name_or_path}")
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        config = {**DEFAULT_EXPERIMENT_CONFIG, **payload}
        config["config_path"] = str(candidate)
        return config

    def available_configs(self) -> List[str]:
        root = Path(self.settings.paths.experiment_config_dir)
        if not root.exists():
            return []
        return sorted(p.stem for p in root.glob("*.y*ml"))

    # ------------------------------------------------------------------
    def run(
        self,
        config: Dict[str, Any],
        dataset: Optional[DemoDataset] = None,
        persist: bool = True,
    ) -> ExperimentResult:
        started = time.perf_counter()
        seed = int(config.get("seed", 42))
        random.seed(seed)
        self.settings.seed = seed

        dataset = dataset or load_demo_dataset(self.settings)
        questions = self._select_questions(dataset, config)
        repetitions = max(1, int(config.get("repetitions", 1)))
        modes = [RunMode(m) for m in config.get("modes", [])]
        sweep = config.get("fault_sweep")

        experiment_id = new_id("exp")
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        label = self.settings.labels.get("results", "DEMO / MOCK RESULTS")

        if persist:
            self.app.repository.create_experiment(
                {
                    "_id": experiment_id,
                    "name": config.get("name", experiment_id),
                    "description": config.get("description", ""),
                    "dataset": config.get("dataset", "demo"),
                    "fault_types": list(sweep) if sweep else ([config["fault_type"]] if config.get("fault_type") else []),
                    "model": f"{self.settings.generation.provider}:{self.settings.generation.model}",
                    "retrieval_config": {**self.settings.retrieval.__dict__, **(config.get("retrieval") or {})},
                    "recovery_config": {**self.settings.budget.__dict__, **(config.get("recovery") or {})},
                    "repetitions": repetitions,
                    "seed": seed,
                    "mode": ",".join(m.value for m in modes),
                    "config_path": config.get("config_path"),
                    "config_hash": config_hash,
                    "status": "running",
                    "result_label": label,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            self.app.repository.save_experiment_config(config.get("name", experiment_id), config, seed)

        records: Dict[str, List[RunMetrics]] = {}
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=config.get("name", experiment_id),
            config=config,
            label=label,
        )

        try:
            plan: List[tuple[RunMode, str]] = [
                (mode, config.get("fault_type")) for mode in modes
            ]
            if sweep:
                plan = [
                    (mode, fault)
                    for mode in modes
                    for fault in sweep
                ]

            for mode, fault_type in plan:
                for repetition in range(repetitions):
                    for question in questions:
                        record, run = self._run_one(
                            question, mode, repetition, fault_type, config, experiment_id
                        )
                        if record is None:
                            continue
                        key = (
                            "mode1" if mode is RunMode.MODE_1_NORMAL_RAG
                            else "mode2" if mode is RunMode.MODE_2_AGENTIC_NO_RECOVERY
                            else "mode3"
                        )
                        records.setdefault(f"{key}:{fault_type or 'none'}", []).append(record)
                        result.run_ids.setdefault(key, []).append(run.run_id)
                        if len(result.run_ids) > 40:  # keep the payload bounded
                            result.run_ids[key] = result.run_ids[key][-400:]

            for mode in modes:
                key = (
                    "mode1" if mode is RunMode.MODE_1_NORMAL_RAG
                    else "mode2" if mode is RunMode.MODE_2_AGENTIC_NO_RECOVERY
                    else "mode3"
                )
                mode_records = [
                    record
                    for bucket, values in records.items()
                    if bucket == key or bucket.startswith(f"{key}:")
                    for record in values
                ]
                if sweep:
                    mode_records = [
                        record
                        for bucket, values in records.items()
                        if bucket.startswith(f"{key}:")
                        for record in values
                    ]
                aggregated = aggregate(
                    mode_records,
                    experiment_id=experiment_id,
                    mode=mode.value,
                    label=label,
                )
                aggregated.failure_count_by_class = {
                    bucket.split(":", 1)[1]: len(values)
                    for bucket, values in records.items()
                    if bucket.startswith(f"{key}:") and bucket.split(":", 1)[1] != "none"
                }
                if mode is RunMode.MODE_1_NORMAL_RAG:
                    result.mode1 = aggregated
                elif mode is RunMode.MODE_2_AGENTIC_NO_RECOVERY:
                    result.mode2 = aggregated
                else:
                    result.mode3 = aggregated
                if persist:
                    self.app.repository.record_metrics(
                        experiment_id=experiment_id,
                        metrics=_numeric_payload(aggregated),
                        group_name=f"summary:{mode.value}",
                        label=label,
                        seed=seed,
                    )

            result.comparison = compare_modes(
                result.mode1 or ExperimentMetrics(),
                result.mode2 or ExperimentMetrics(),
                result.mode3 or ExperimentMetrics(),
            )
        except Exception as exc:
            logger.exception("experiment %s failed", experiment_id)
            if persist:
                self.app.repository.update_experiment(
                    experiment_id, {"status": "failed", "error": str(exc)}
                )
            raise

        result.duration_s = time.perf_counter() - started
        if persist:
            self.app.repository.update_experiment(
                experiment_id,
                {
                    "status": "completed",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": None,
                },
            )
        return result

    # ------------------------------------------------------------------
    def _run_one(
        self,
        question: DemoQuestion,
        mode: RunMode,
        repetition: int,
        fault_type: Optional[str],
        config: Dict[str, Any],
        experiment_id: str,
    ) -> tuple[Optional[RunMetrics], PipelineRun]:
        faults: List[Any] = []
        if fault_type and mode is RunMode.MODE_3_AGENTIC_WITH_RECOVERY:
            faults = [fault_type]
        ground_truth = {}
        if faults:
            definition = self.app.orchestrator.faults.definition(str(faults[0]))
            if definition is not None:
                ground_truth = {
                    "expected_failure_class": definition.expected_failure_class.value,
                    "expected_subtype": definition.expected_subtype,
                    "expected_symptom": definition.expected_symptom,
                    "fault_type": definition.fault_type,
                }
        try:
            outcome = self.app.orchestrator.answer(
                question.question,
                mode=mode,
                faults=faults,
                fault_parameters=config.get("fault_parameters") or {},
                experiment_id=experiment_id,
                relevant_documents=question.relevant_documents,
                answer_keywords=question.answer_keywords,
                labels={"results": self.settings.labels.get("results", "DEMO / MOCK RESULTS"),
                        "repetition": str(repetition)},
            )
        except Exception as exc:  # a failed run is data, not a crash
            logger.warning("run failed for %s: %s", question.question_id, exc)
            return None, PipelineRun(query=question.question)

        record = evaluate_run(
            outcome.run,
            state=outcome.state,
            reference_answer=question.answer,
            answer_keywords=question.answer_keywords,
            ground_truth=ground_truth,
            label=self.settings.labels.get("results", ""),
        )
        return record, outcome.run

    def _select_questions(self, dataset: DemoDataset, config: Dict[str, Any]) -> List[DemoQuestion]:
        ids = config.get("question_ids")
        questions = dataset.questions
        if ids:
            wanted = set(ids)
            questions = [q for q in questions if q.question_id in wanted]
        category = config.get("category")
        if category:
            questions = [q for q in questions if q.category == category]
        limit = config.get("limit")
        if limit:
            questions = questions[: int(limit)]
        return questions


def _numeric_payload(metrics: ExperimentMetrics) -> Dict[str, Any]:
    """Flatten an ExperimentMetrics into JSON-serialisable metric values."""
    payload = metrics.model_dump()
    payload.pop("per_query", None)
    payload.pop("created_at", None)
    payload["notes"] = "; ".join(metrics.notes)
    return payload


def run_experiment_by_name(
    app: Application, name: str, **overrides: Any
) -> ExperimentResult:
    runner = ExperimentRunner(app)
    config = runner.load_config(name)
    config.update({k: v for k, v in overrides.items() if v is not None})
    return runner.run(config)


__all__ = [
    "ExperimentRunner",
    "ExperimentResult",
    "run_experiment_by_name",
    "DEFAULT_EXPERIMENT_CONFIG",
    "compare_modes",
    "evaluate_run",
    "aggregate",
]