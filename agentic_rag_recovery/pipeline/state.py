"""Shared pipeline state.

The state object is deliberately explicit (not hidden inside a framework) so
that the same state works with the LangGraph runner and the sequential
runner, and so recovery actions can surgically re-run a single stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..models import (
    Abstention,
    AgentStep,
    AssembledContext,
    BudgetSnapshot,
    ContextItem,
    Diagnosis,
    GenerationResult,
    PipelineRun,
    QueryUnderstanding,
    RecoveryAttempt,
    RetrievedChunk,
    RetrievalDiagnostics,
    RunMode,
    RunStatus,
    SignalBundle,
    new_id,
)
from ..taxonomy import RecoveryAction, Stage


@dataclass
class SubQueryResult:
    """Per-sub-query retrieval outcome (agentic branch bookkeeping)."""

    subquery: str
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    diagnostics: Optional[RetrievalDiagnostics] = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subquery": self.subquery,
            "count": len(self.retrieved),
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
            "top_chunk": self.retrieved[0].chunk_id if self.retrieved else None,
            "top_score": self.retrieved[0].score if self.retrieved else 0.0,
            "useful_ratio": self.diagnostics.useful_ratio if self.diagnostics else 0.0,
            "chunk_ids": [c.chunk_id for c in self.retrieved],
        }


@dataclass
class StageResult:
    """Result of executing (or re-executing) one stage."""

    stage: Stage
    ok: bool = True
    latency_ms: float = 0.0
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    reused: bool = False


@dataclass
class InfraCounters:
    timeouts: int = 0
    infra_errors: int = 0
    provider_errors: int = 0
    parse_errors: int = 0
    embedding_errors: int = 0
    retrieval_service_unavailable: bool = False
    service_status: str = "ok"
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)

    def add(self, stage: Stage, latency_ms: float) -> None:
        key = stage.value if isinstance(stage, Stage) else str(stage)
        self.stage_latency_ms[key] = round(self.stage_latency_ms.get(key, 0.0) + latency_ms, 3)

    def as_dict(self, total_latency_ms: float = 0.0) -> Dict[str, Any]:
        return {
            "timeouts": self.timeouts,
            "infra_errors": self.infra_errors,
            "provider_errors": self.provider_errors,
            "parse_errors": self.parse_errors,
            "embedding_errors": self.embedding_errors,
            "retrieval_service_unavailable": self.retrieval_service_unavailable,
            "service_status": self.service_status,
            "stage_latency_ms": dict(self.stage_latency_ms),
            "latency_ms": round(total_latency_ms, 3),
        }


@dataclass
class PipelineState:
    """Everything one request needs, plus the history of what was re-run."""

    run_id: str = field(default_factory=lambda: new_id("run"))
    experiment_id: Optional[str] = None
    mode: RunMode = RunMode.MODE_3_AGENTIC_WITH_RECOVERY
    query: str = ""

    understanding: Optional[QueryUnderstanding] = None
    sub_results: List[SubQueryResult] = field(default_factory=list)
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    retrieval_diagnostics: Optional[RetrievalDiagnostics] = None
    retrieved_candidates: List[RetrievedChunk] = field(default_factory=list)
    context: Optional[AssembledContext] = None
    generation: Optional[GenerationResult] = None
    signals: Optional[SignalBundle] = None

    diagnoses: List[Diagnosis] = field(default_factory=list)
    recoveries: List[RecoveryAttempt] = field(default_factory=list)
    abstention: Optional[Abstention] = None
    stage_results: List[StageResult] = field(default_factory=list)
    executed_stages: List[Stage] = field(default_factory=list)
    skipped_stages: List[Stage] = field(default_factory=list)

    infra: InfraCounters = field(default_factory=InfraCounters)
    llm_calls: int = 0
    tokens_used: int = 0
    cost: float = 0.0
    total_latency_ms: float = 0.0
    baseline_answer: str = ""
    baseline_latency_ms: float = 0.0
    baseline_generation: Optional[GenerationResult] = None
    baseline_signals: Optional[SignalBundle] = None

    active_faults: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth: Optional[Dict[str, Any]] = None
    gold_chunk_ids: Set[str] = field(default_factory=set)
    gold_document_ids: Set[str] = field(default_factory=set)
    aborted: bool = False
    abort_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def mark_executed(self, stage: Stage) -> None:
        if stage not in self.executed_stages:
            self.executed_stages.append(stage)

    def record_stage(self, result: StageResult) -> None:
        self.stage_results.append(result)
        if result.ok:
            self.mark_executed(result.stage)
        self.infra.add(result.stage, result.latency_ms)
        self.total_latency_ms = round(self.total_latency_ms + result.latency_ms, 3)

    def add_llm(self, calls: int = 1, tokens: int = 0, cost: float = 0.0) -> None:
        self.llm_calls += int(calls)
        self.tokens_used += int(tokens)
        self.cost += float(cost)

    def attempted_actions(self) -> List[RecoveryAction]:
        return [a.action for a in self.recoveries]

    def answer(self) -> str:
        return self.generation.answer if self.generation else ""

    def final_diagnosis(self) -> Optional[Diagnosis]:
        return self.diagnoses[-1] if self.diagnoses else None

    def recovered(self) -> bool:
        return any(r.status.value == "success" for r in self.recoveries)

    def snapshot_for_recovery(self) -> Dict[str, Any]:
        """Compact snapshot used as ``before`` in recovery records."""
        return {
            "retrieved": len(self.retrieved),
            "top_chunk": self.retrieved[0].chunk_id if self.retrieved else None,
            "top_score": self.retrieved[0].score if self.retrieved else 0.0,
            "useful_ratio": self.retrieval_diagnostics.useful_ratio if self.retrieval_diagnostics else 0.0,
            "context_tokens": self.context.token_estimate if self.context else 0,
            "context_coverage": self.context.coverage if self.context else 0.0,
            "answer": (self.answer() or "")[:400],
            "groundedness": self.signals.generation.get("groundedness") if self.signals else None,
        }

    # ------------------------------------------------------------------
    def to_run(self, labels: Optional[Dict[str, str]] = None) -> PipelineRun:
        status = RunStatus.SUCCESS
        if self.abstention is not None and self.abstention.abstained:
            status = RunStatus.ABSTAINED
        elif self.diagnoses and self.diagnoses[0].failure_detected:
            status = RunStatus.RECOVERED if self.recovered() else RunStatus.FAILED
        run = PipelineRun(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            mode=self.mode,
            status=status,
            query=self.query,
            labels=labels or {},
            understanding=self.understanding,
            sub_results=[r.to_dict() for r in self.sub_results],
            retrieved=self.retrieved,
            retrieval_diagnostics=self.retrieval_diagnostics,
            context=self.context,
            generation=self.generation,
            signals=self.signals,
            diagnoses=self.diagnoses,
            recoveries=self.recoveries,
            abstention=self.abstention,
            answer=self.answer(),
            baseline_answer=self.baseline_answer,
            baseline_latency_ms=self.baseline_latency_ms,
            total_latency_ms=self.total_latency_ms,
            added_latency_ms=round(max(0.0, self.total_latency_ms - self.baseline_latency_ms), 3),
            tokens_used=self.tokens_used,
            llm_calls=self.llm_calls,
            cost=self.cost,
            faults=self.active_faults,
        )
        return run

    def to_budget_signal(self, snapshot: BudgetSnapshot) -> Dict[str, Any]:
        payload = snapshot.model_dump()
        payload["exhausted"] = snapshot.exhausted()
        payload["first_exhausted"] = snapshot.first_exhausted()
        return payload