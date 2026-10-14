"""Core domain models shared across the pipeline, diagnosis and recovery."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .taxonomy import FailureClass, RecoveryAction, Severity, Stage


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_hash(*parts: Any) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class Model(BaseModel):
    model_config = ConfigDict(use_enum_values=False, extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class Document(Model):
    document_id: str = Field(default_factory=lambda: new_id("doc"))
    source: str
    title: str = ""
    doc_type: str = "txt"
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: str = ""
    version: int = 1
    is_active: bool = True
    created_at: str = Field(default_factory=utcnow)


class Chunk(Model):
    chunk_id: str = Field(default_factory=lambda: new_id("chk"))
    document_id: str
    chunk_index: int = 0
    content: str = ""
    token_estimate: int = 0
    char_start: int = 0
    char_end: int = 0
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    created_at: str = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class RetrievalMethod(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    NONE = "none"


class RetrievedChunk(Model):
    chunk_id: str
    document_id: str
    content: str
    score: float = 0.0
    rank: int = 0
    method: str = RetrievalMethod.VECTOR.value
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None
    page: Optional[int] = None
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_relevant: Optional[bool] = None  # only when ground truth is available
    truncated: bool = False
    expanded_from: Optional[str] = None

    def key(self) -> str:
        return self.chunk_id


class RetrievalDiagnostics(Model):
    """Diagnostics that make retrieval behaviour observable."""

    top_k: int = 0
    candidate_pool: int = 0
    returned: int = 0
    top_score: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    score_gap: float = 0.0
    score_distribution: List[float] = Field(default_factory=list)
    score_std: float = 0.0
    top_overlap: float = 0.0
    mean_overlap: float = 0.0
    pre_rank_top_overlap: float = 0.0
    pre_rank_useful_ratio: float = 0.0
    ranking_coverage_loss: float = 0.0
    candidate_chunk_ids: List[str] = Field(default_factory=list)
    useful_overlap: float = 0.0
    useful_count: int = 0
    useful_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    source_diversity: int = 0
    unique_documents: int = 0
    latency_ms: float = 0.0
    method: str = RetrievalMethod.NONE.value
    rank_changes: int = 0
    reranker_applied: bool = False
    rerank_score_margin: float = 0.0
    notes: List[str] = Field(default_factory=list)

    def top_ratio(self) -> float:
        return self.useful_ratio


# ---------------------------------------------------------------------------
# Agentic workflow
# ---------------------------------------------------------------------------
class QueryUnderstanding(Model):
    original_query: str
    normalized_query: str = ""
    rewrite: Optional[str] = None
    subqueries: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    is_complex: bool = False
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    intent: str = "factual_lookup"
    language: str = "en"
    rewrite_drift: float = 0.0
    notes: List[str] = Field(default_factory=list)


class ContextItem(Model):
    chunk: RetrievedChunk
    text: str
    position: int = 0
    truncated: bool = False
    expansion_reason: Optional[str] = None


class AssembledContext(Model):
    items: List[ContextItem] = Field(default_factory=list)
    text: str = ""
    token_estimate: int = 0
    char_estimate: int = 0
    source_diversity: int = 0
    unique_documents: int = 0
    fragmentation_ratio: float = 0.0
    conflict_score: float = 0.0
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    truncation_applied: bool = False
    coverage: float = 0.0
    unconstrained_constraints: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class GenerationResult(Model):
    answer: str = ""
    raw_answer: str = ""
    citations: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    grounded: bool = True
    groundedness: float = 1.0
    citation_coverage: float = 0.0
    unsupported_claims: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    grounded_mode: bool = True
    provider: str = "mock"
    model: str = "mock"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    latency_budget_ok: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)


class StageTiming(Model):
    stage: str
    latency_ms: float = 0.0
    status: str = "ok"
    input_summary: str = ""
    output_summary: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


class AgentStep(Model):
    step_id: str = Field(default_factory=lambda: new_id("stp"))
    index: int = 0
    stage: str
    action: str
    input_summary: str = ""
    output_summary: str = ""
    status: str = "ok"
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=utcnow)
    tool_name: Optional[str] = None
    action_signature: str = ""
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
class DiagnosisTier(str, Enum):
    NONE = "none"
    RULE = "rule"
    LLM = "llm"
    FALLBACK = "fallback"
    GROUND_TRUTH = "ground_truth"


class Diagnosis(Model):
    diagnosis_id: str = Field(default_factory=lambda: new_id("dg"))
    run_id: str = ""
    failure_detected: bool = False
    failure_class: FailureClass = FailureClass.NONE
    stage: Optional[Stage] = None
    subtype: Optional[str] = None
    severity: Severity = Severity.LOW
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    recommended_recovery: Optional[RecoveryAction] = None
    recovery_ladder: List[RecoveryAction] = Field(default_factory=list)
    tier: DiagnosisTier = DiagnosisTier.NONE
    rule_id: Optional[str] = None
    rule_matches: List[Dict[str, Any]] = Field(default_factory=list)
    signals_used: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    raw_llm_output: Optional[str] = None
    validation_errors: List[str] = Field(default_factory=list)
    repairs: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def class_value(self) -> str:
        return self.failure_class.value if isinstance(self.failure_class, FailureClass) else str(self.failure_class)

    def summary(self) -> Dict[str, Any]:
        return {
            "diagnosis_id": self.diagnosis_id,
            "failure_detected": self.failure_detected,
            "failure_class": self.class_value,
            "stage": self.stage.value if self.stage else None,
            "subtype": self.subtype,
            "severity": self.severity.value if isinstance(self.severity, Severity) else str(self.severity),
            "confidence": round(self.confidence, 4),
            "tier": self.tier.value if isinstance(self.tier, DiagnosisTier) else str(self.tier),
            "recommended_recovery": (
                self.recommended_recovery.value
                if isinstance(self.recommended_recovery, RecoveryAction)
                else self.recommended_recovery
            ),
            "evidence": self.evidence,
        }


class SignalBundle(Model):
    """Full set of runtime signals collected for one pipeline execution."""

    run_id: str = ""
    query: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    ranking: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    generation: Dict[str, Any] = Field(default_factory=dict)
    agent: Dict[str, Any] = Field(default_factory=dict)
    infrastructure: Dict[str, Any] = Field(default_factory=dict)
    budget: Dict[str, Any] = Field(default_factory=dict)
    collected_at: str = Field(default_factory=utcnow)

    def flat(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for section in self.model_dump().values():
            if isinstance(section, dict):
                for key, value in section.items():
                    if not isinstance(value, (list, dict)):
                        out[key] = value
        return out

    def get(self, key: str, default: Any = None) -> Any:
        return self.flat().get(key, default)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------
class RecoveryOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    ABSTAIN = "abstain"
    ERROR = "error"


class RecoveryAttempt(Model):
    attempt_id: str = Field(default_factory=lambda: new_id("rec"))
    run_id: str = ""
    attempt_index: int = 0
    action: RecoveryAction
    target_stage: Stage
    diagnosis_subtype: Optional[str] = None
    status: RecoveryOutcome = RecoveryOutcome.FAILURE
    rationale: str = ""
    reused_stage: bool = False
    skipped_stages: List[str] = Field(default_factory=list)
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)
    verification: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    added_latency_ms: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    llm_calls: int = 0
    error: Optional[str] = None
    created_at: str = Field(default_factory=utcnow)


class Abstention(Model):
    abstained: bool = False
    reason: str = ""
    failure_class: FailureClass = FailureClass.NONE
    stage: Optional[Stage] = None
    subtype: Optional[str] = None
    attempted_recoveries: List[RecoveryAction] = Field(default_factory=list)
    attempts_used: int = 0
    max_attempts: int = 0
    budget_consumed: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    created_at: str = Field(default_factory=utcnow)


class BudgetSnapshot(Model):
    max_recovery_attempts: int = 0
    max_total_latency_ms: int = 0
    max_llm_calls: int = 0
    max_token_budget: int = 0
    max_repeated_actions: int = 1

    attempts_used: int = 0
    attempts_remaining: int = 0
    latency_used_ms: float = 0.0
    latency_remaining_ms: float = 0.0
    llm_calls_used: int = 0
    llm_calls_remaining: int = 0
    tokens_used: int = 0
    tokens_remaining: int = 0
    repeated_actions_blocked: int = 0

    def exhausted(self) -> Dict[str, bool]:
        return {
            "attempts": self.attempts_remaining <= 0,
            "latency": self.latency_remaining_ms <= 0,
            "llm_calls": self.llm_calls_remaining <= 0,
            "tokens": self.tokens_remaining <= 0,
        }

    def first_exhausted(self) -> Optional[str]:
        for name, flag in self.exhausted().items():
            if flag:
                return name
        return None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
class RunMode(str, Enum):
    MODE_1_NORMAL_RAG = "mode1_normal_rag"
    MODE_2_AGENTIC_NO_RECOVERY = "mode2_agentic_no_recovery"
    MODE_3_AGENTIC_WITH_RECOVERY = "mode3_agentic_with_recovery"


class RunStatus(str, Enum):
    SUCCESS = "success"
    RECOVERED = "recovered"
    FAILED = "failed"
    ABSTAINED = "abstained"
    ERROR = "error"


class TraceSpan(Model):
    span_id: str = Field(default_factory=lambda: new_id("spn"))
    parent_id: Optional[str] = None
    run_id: str = ""
    stage: str
    name: str = ""
    start_time: str = Field(default_factory=utcnow)
    end_time: Optional[str] = None
    latency_ms: float = 0.0
    status: str = "ok"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class PipelineRun(Model):
    """One end-to-end request through the agentic pipeline."""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    experiment_id: Optional[str] = None
    mode: RunMode = RunMode.MODE_3_AGENTIC_WITH_RECOVERY
    status: RunStatus = RunStatus.SUCCESS
    query: str = ""
    labels: Dict[str, str] = Field(default_factory=dict)
    understanding: Optional[QueryUnderstanding] = None
    sub_results: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_diagnostics: Optional[RetrievalDiagnostics] = None
    context: Optional[AssembledContext] = None
    generation: Optional[GenerationResult] = None
    signals: Optional[SignalBundle] = None
    diagnoses: List[Diagnosis] = Field(default_factory=list)
    recoveries: List[RecoveryAttempt] = Field(default_factory=list)
    abstention: Optional[Abstention] = None
    answer: str = ""
    baseline_answer: str = ""
    baseline_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    added_latency_ms: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    cost: float = 0.0
    steps: List[AgentStep] = Field(default_factory=list)
    spans: List[TraceSpan] = Field(default_factory=list)
    faults: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)
    finished_at: Optional[str] = None

    @property
    def final_diagnosis(self) -> Optional[Diagnosis]:
        return self.diagnoses[-1] if self.diagnoses else None

    @property
    def recovered(self) -> bool:
        return any(a.status == RecoveryOutcome.SUCCESS for a in self.recoveries)

    @property
    def abstained(self) -> bool:
        return bool(self.abstention and self.abstention.abstained)


class FaultInjection(Model):
    """A deliberately injected fault plus its ground-truth label."""

    fault_id: str = Field(default_factory=lambda: new_id("flt"))
    fault_type: str
    target_stage: Stage
    severity: Severity = Severity.MEDIUM
    description: str = ""
    expected_symptom: str = ""
    expected_failure_class: FailureClass = FailureClass.INFRASTRUCTURE_FAILURE
    expected_subtype: Optional[str] = None
    injection_parameters: Dict[str, Any] = Field(default_factory=dict)
    active: bool = False
    experiment_id: Optional[str] = None
    created_at: str = Field(default_factory=utcnow)

    def ground_truth(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type,
            "target_stage": self.target_stage.value,
            "expected_failure_class": self.expected_failure_class.value,
            "expected_subtype": self.expected_subtype,
            "severity": self.severity.value,
            "expected_symptom": self.expected_symptom,
        }


class ExperimentMetrics(Model):
    """Aggregated, honestly-labelled metrics for one experiment."""

    experiment_id: str = ""
    label: str = "DEMO / MOCK RESULTS"
    mode: str = ""
    total_runs: int = 0
    diagnosable_failures: int = 0
    correct_diagnoses: int = 0
    diagnosis_accuracy: float = 0.0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    recovery_success_rate: float = 0.0
    groundedness: float = 0.0
    baseline_groundedness: float = 0.0
    added_latency_ms: float = 0.0
    cost_per_recovery: float = 0.0
    abstentions: int = 0
    abstention_rate: float = 0.0
    answer_correctness: float = 0.0
    baseline_answer_correctness: float = 0.0
    retrieval_recall: float = 0.0
    baseline_retrieval_recall: float = 0.0
    retrieval_precision: float = 0.0
    mrr: float = 0.0
    context_coverage: float = 0.0
    citation_coverage: float = 0.0
    avg_recovery_attempts: float = 0.0
    token_usage: int = 0
    total_cost: float = 0.0
    failure_count_by_class: Dict[str, int] = Field(default_factory=dict)
    failure_count_by_stage: Dict[str, int] = Field(default_factory=dict)
    severity_distribution: Dict[str, int] = Field(default_factory=dict)
    rule_diagnoses: int = 0
    llm_diagnoses: int = 0
    fallback_diagnoses: int = 0
    diagnosis_latency_ms: float = 0.0
    per_query: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    notes: List[str] = Field(default_factory=list)