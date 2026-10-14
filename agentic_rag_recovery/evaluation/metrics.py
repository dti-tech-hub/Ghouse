"""Evaluation metrics.

All numbers written to SQLite come from these functions, so results are
reproducible offline.  External libraries (RAGAS / DeepEval) are optional
adapters in :mod:`agentic_rag_recovery.evaluation.external` and are never the
source of the reported figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import Diagnosis, ExperimentMetrics, PipelineRun, RecoveryAttempt
from ..signals.metrics import answer_correctness, mean
from ..models import DiagnosisTier


def diagnosis_correct(diagnosis: Optional[Diagnosis], ground_truth: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """Return ``(subtype_correct, class_correct, acceptable)``.

    ``acceptable`` additionally accepts any subtype the fault declared it could
    surface as (a fault has one expected subtype but may legitimately present
    as a neighbouring one).  All three numbers are reported separately - none
    of them is presented as a substitute for another.
    """
    if not ground_truth or diagnosis is None:
        return False, False, False
    expected_subtype = ground_truth.get("expected_subtype")
    expected_class = ground_truth.get("expected_failure_class")
    got_subtype = diagnosis.subtype if diagnosis.failure_detected else None
    got_class = diagnosis.failure_class.value if diagnosis.failure_detected else None
    subtype_ok = bool(expected_subtype) and got_subtype == expected_subtype
    class_ok = bool(expected_class) and got_class == expected_class
    return subtype_ok, class_ok, subtype_ok or class_ok


@dataclass
class RunMetrics:
    """Per-run evaluation record."""

    run_id: str
    mode: str
    status: str
    query: str
    groundedness: float = 0.0
    baseline_groundedness: float = 0.0
    answer_f1: float = 0.0
    baseline_answer_f1: float = 0.0
    keyword_coverage: float = 0.0
    citation_coverage: float = 0.0
    context_coverage: float = 0.0
    recall: Optional[float] = None
    precision: Optional[float] = None
    mrr: Optional[float] = None
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    recovery_success: bool = False
    abstained: bool = False
    diagnosis_detected: bool = False
    diagnosis_subtype_correct: bool = False
    diagnosis_class_correct: bool = False
    diagnosis_acceptable: bool = False
    diagnosis_confidence: float = 0.0
    diagnosis_tier: str = "none"
    total_latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0
    added_latency_ms: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    cost: float = 0.0
    faults: List[str] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def evaluate_run(
    run: PipelineRun,
    state: Any = None,
    reference_answer: str = "",
    answer_keywords: Sequence[str] = (),
    ground_truth: Optional[Dict[str, Any]] = None,
    label: str = "",
) -> RunMetrics:
    """Evaluate one :class:`PipelineRun` against its ground truth."""
    signals = run.signals
    metrics = RunMetrics(
        run_id=run.run_id,
        mode=run.mode.value,
        status=run.status.value,
        query=run.query,
        label=label,
        faults=[f.get("fault_type", "?") for f in (run.faults or [])],
    )
    if signals is not None:
        metrics.groundedness = float(signals.generation.get("groundedness", 0.0) or 0.0)
        metrics.citation_coverage = float(signals.generation.get("citation_coverage", 0.0) or 0.0)
        metrics.context_coverage = float(signals.context.get("context_coverage", 0.0) or 0.0)
        for key in ("recall", "precision", "mrr"):
            value = signals.retrieval.get(key)
            if value is not None:
                setattr(metrics, key, float(value))
    baseline = getattr(state, "baseline_signals", None)
    if baseline is not None:
        metrics.baseline_groundedness = float(baseline.generation.get("groundedness", 0.0) or 0.0)

    if reference_answer:
        correctness = answer_correctness(run.answer, reference_answer, answer_keywords)
        metrics.answer_f1 = correctness["f1"]
        metrics.keyword_coverage = correctness["keyword_coverage"]
        if run.baseline_answer:
            base = answer_correctness(run.baseline_answer, reference_answer, answer_keywords)
            metrics.baseline_answer_f1 = base["f1"]

    # diagnosis is judged on the *first* (pre-recovery) diagnosis
    diagnosis = run.diagnoses[0] if run.diagnoses else None
    if diagnosis is not None:
        metrics.diagnosis_detected = diagnosis.failure_detected
        metrics.diagnosis_confidence = diagnosis.confidence
        metrics.diagnosis_tier = diagnosis.tier.value if hasattr(diagnosis.tier, "value") else str(diagnosis.tier)
        subtype_ok, class_ok, acceptable = diagnosis_correct(diagnosis, ground_truth or {})
        metrics.diagnosis_subtype_correct = subtype_ok
        metrics.diagnosis_class_correct = class_ok
        metrics.diagnosis_acceptable = acceptable

    metrics.recovery_attempts = len(run.recoveries)
    metrics.successful_recoveries = sum(
        1 for a in run.recoveries if a.status.value == "success"
    )
    metrics.recovery_success = metrics.successful_recoveries > 0
    metrics.abstained = run.abstained
    metrics.total_latency_ms = run.total_latency_ms
    metrics.baseline_latency_ms = run.baseline_latency_ms
    metrics.added_latency_ms = run.added_latency_ms
    metrics.tokens_used = run.tokens_used
    metrics.llm_calls = run.llm_calls
    metrics.cost = run.cost
    return metrics


def aggregate(
    records: Sequence[RunMetrics],
    experiment_id: str = "",
    mode: str = "",
    label: str = "DEMO / MOCK RESULTS",
) -> ExperimentMetrics:
    """Aggregate per-run records into the reported metric bundle."""
    out = ExperimentMetrics(experiment_id=experiment_id, mode=mode, label=label)
    out.total_runs = len(records)
    if not records:
        return out

    diagnosable = [r for r in records if r.faults and r.diagnosis_detected]
    out.diagnosable_failures = len(diagnosable)
    out.correct_diagnoses = sum(1 for r in diagnosable if r.diagnosis_subtype_correct)
    out.diagnosis_accuracy = (
        out.correct_diagnoses / out.diagnosable_failures if out.diagnosable_failures else 0.0
    )

    attempts = sum(r.recovery_attempts for r in records)
    successes = sum(r.successful_recoveries for r in records)
    out.recovery_attempts = attempts
    out.successful_recoveries = successes
    out.recovery_success_rate = successes / attempts if attempts else 0.0

    out.groundedness = round(mean([r.groundedness for r in records]), 6)
    out.baseline_groundedness = round(mean([r.baseline_groundedness for r in records]), 6)
    out.added_latency_ms = round(mean([r.added_latency_ms for r in records]), 3)
    out.cost_per_recovery = round(
        sum(r.cost for r in records) / successes if successes else 0.0, 8
    )
    out.abstentions = sum(1 for r in records if r.abstained)
    out.abstention_rate = out.abstentions / out.total_runs
    out.answer_correctness = round(mean([r.answer_f1 for r in records]), 6)
    out.baseline_answer_correctness = round(mean([r.baseline_answer_f1 for r in records]), 6)
    out.retrieval_recall = round(mean([r.recall for r in records if r.recall is not None]), 6)
    out.baseline_retrieval_recall = out.retrieval_recall
    out.retrieval_precision = round(mean([r.precision for r in records if r.precision is not None]), 6)
    out.mrr = round(mean([r.mrr for r in records if r.mrr is not None]), 6)
    out.context_coverage = round(mean([r.context_coverage for r in records]), 6)
    out.citation_coverage = round(mean([r.citation_coverage for r in records]), 6)
    out.avg_recovery_attempts = round(attempts / out.total_runs, 4)
    out.token_usage = sum(r.tokens_used for r in records)
    out.total_cost = round(sum(r.cost for r in records), 8)

    by_class: Dict[str, int] = {}
    by_stage: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for record in records:
        for fault in record.faults:
            by_class[fault] = by_class.get(fault, 0) + 1
    out.failure_count_by_class = by_class
    out.failure_count_by_stage = {}
    out.severity_distribution = by_severity

    out.rule_diagnoses = sum(1 for r in records if r.diagnosis_tier == DiagnosisTier.RULE.value)
    out.llm_diagnoses = sum(1 for r in records if r.diagnosis_tier == DiagnosisTier.LLM.value)
    out.fallback_diagnoses = sum(1 for r in records if r.diagnosis_tier == DiagnosisTier.FALLBACK.value)
    out.per_query = [r.to_dict() for r in records]
    return out


def compare_modes(
    mode1: ExperimentMetrics, mode2: ExperimentMetrics, mode3: ExperimentMetrics
) -> Dict[str, Any]:
    """Side-by-side comparison of the three experimental modes.

    Only deltas between *actually executed* runs are reported.
    """
    def block(m: ExperimentMetrics) -> Dict[str, Any]:
        return {
            "runs": m.total_runs,
            "groundedness": m.groundedness,
            "answer_correctness": m.answer_correctness,
            "retrieval_recall": m.retrieval_recall,
            "context_coverage": m.context_coverage,
            "citation_coverage": m.citation_coverage,
            "abstention_rate": m.abstention_rate,
            "avg_recovery_attempts": m.avg_recovery_attempts,
            "added_latency_ms": m.added_latency_ms,
            "diagnosis_accuracy": m.diagnosis_accuracy,
            "recovery_success_rate": m.recovery_success_rate,
            "cost_per_recovery": m.cost_per_recovery,
            "total_cost": m.total_cost,
            "token_usage": m.token_usage,
        }

    b1, b2, b3 = block(mode1), block(mode2), block(mode3)
    delta = {
        key: round(float(b3.get(key, 0.0)) - float(b1.get(key, 0.0)), 6)
        for key in ("groundedness", "answer_correctness", "abstention_rate", "added_latency_ms")
        if isinstance(b1.get(key), (int, float)) and isinstance(b3.get(key), (int, float))
    }
    return {
        "mode1_normal_rag": b1,
        "mode2_agentic_no_recovery": b2,
        "mode3_agentic_with_recovery": b3,
        "mode3_minus_mode1": delta,
        "note": (
            "Mode 1 and Mode 2 run without fault injection; Mode 3 runs on the "
            "same questions with the declared fault active."
        ),
    }


def severity_distribution(diagnoses: Iterable[Diagnosis]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for diagnosis in diagnoses:
        if not diagnosis.failure_detected:
            continue
        key = diagnosis.severity.value if hasattr(diagnosis.severity, "value") else str(diagnosis.severity)
        out[key] = out.get(key, 0) + 1
    return out


def stage_distribution(diagnoses: Iterable[Diagnosis]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for diagnosis in diagnoses:
        if not diagnosis.failure_detected or diagnosis.stage is None:
            continue
        key = diagnosis.stage.value if hasattr(diagnosis.stage, "value") else str(diagnosis.stage)
        out[key] = out.get(key, 0) + 1
    return out


def class_distribution(diagnoses: Iterable[Diagnosis]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for diagnosis in diagnoses:
        if not diagnosis.failure_detected:
            continue
        key = (
            diagnosis.failure_class.value
            if hasattr(diagnosis.failure_class, "value")
            else str(diagnosis.failure_class)
        )
        out[key] = out.get(key, 0) + 1
    return out